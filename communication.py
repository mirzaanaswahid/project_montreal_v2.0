#!/usr/bin/env python3
"""
communication.py - Inter-agent messaging system for EAGLE
Handles all agent-to-agent communications with realistic delays and logging
"""

import time
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import numpy as np


class MessageType(Enum):
    """EAGLE message types from the formulation"""
    # Event-related
    HANDOVER_REQUEST = "HANDOVER_REQUEST"
    EVENT_BID = "EVENT_BID"
    TASK_CLAIMED = "TASK_CLAIMED"
    INVESTIGATING_LAST_RESORT = "INVESTIGATING_LAST_RESORT"
    
    # Thermal-related
    THERMAL_DISCOVERED = "THERMAL_DISCOVERED"
    LIVE_THERMAL_BID = "LIVE_THERMAL_BID"
    THERMAL_CLAIMED = "THERMAL_CLAIMED"
    
    # Coordination
    PATROL_HANDOVER_REQUEST = "PATROL_HANDOVER_REQUEST"
    PATROL_HANDOVER_RESPONSE = "PATROL_HANDOVER_RESPONSE"
    
    # State sharing
    AGENT_STATE = "AGENT_STATE"
    HEARTBEAT = "HEARTBEAT"


@dataclass
class Message:
    """Standard message format for EAGLE communications"""
    msg_id: str
    msg_type: MessageType
    sender_id: str
    timestamp: float
    data: Dict[str, Any]
    ttl: float = 30.0  # Time to live in seconds
    recipients: Optional[Set[str]] = None  # None = broadcast
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "ttl": self.ttl,
            "recipients": list(self.recipients) if self.recipients else None
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Message':
        """Create from dictionary"""
        return cls(
            msg_id=d["msg_id"],
            msg_type=MessageType(d["msg_type"]),
            sender_id=d["sender_id"],
            timestamp=d["timestamp"],
            data=d["data"],
            ttl=d.get("ttl", 30.0),
            recipients=set(d["recipients"]) if d.get("recipients") else None
        )


class CommunicationNetwork:
    """
    Simulates realistic UAV communication network with:
    - Range-based connectivity
    - Message delays
    - Packet loss
    - Message history for debugging
    """
    
    def __init__(self, 
                 comm_range: float = 5000.0,  # meters
                 base_delay: float = 0.1,      # seconds
                 packet_loss_rate: float = 0.02,
                 enable_logging: bool = True):
        
        self.comm_range = comm_range
        self.base_delay = base_delay
        self.packet_loss_rate = packet_loss_rate
        
        # Message queues for each agent
        self.inbox: Dict[str, deque] = defaultdict(deque)
        self.outbox: Dict[str, List[Message]] = defaultdict(list)
        
        # Agent positions for range checking
        self.agent_positions: Dict[str, np.ndarray] = {}
        
        # Message tracking
        self.message_history: List[Message] = []
        self.message_counter = 0
        
        # Simulation clock
        self.sim_time: float = 0.0
        
        # Logging
        if enable_logging:
            self.logger = logging.getLogger("EAGLE_Comms")
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = None
    
    def register_agent(self, agent_id: str, position: np.ndarray):
        """Register an agent with the network"""
        self.agent_positions[agent_id] = position.copy()
        if agent_id not in self.inbox:
            self.inbox[agent_id] = deque()
        self._log(f"Agent {agent_id} registered at position {position}")
    
    def update_agent_position(self, agent_id: str, position: np.ndarray):
        """Update agent position for range calculations"""
        self.agent_positions[agent_id] = position.copy()
    
    def set_sim_time(self, t: float) -> None:
        """Set the current simulation time (seconds)."""
        self.sim_time = float(t)
    
    def send_message(self, msg: Message):
        """Queue a message for delivery"""
        self.message_counter += 1
        msg.msg_id = f"{msg.sender_id}_{self.message_counter:06d}"
        
        # Store in history
        self.message_history.append(msg)
        
        # Add to outbox for processing
        self.outbox[msg.sender_id].append(msg)
        
        self._log(f"Message queued: {msg.msg_type.value} from {msg.sender_id}")
    
    def broadcast(self, sender_id: str, msg_type: MessageType, data: Dict[str, Any]):
        """Convenience method for broadcasting"""
        msg = Message(
            msg_id="",
            msg_type=msg_type,
            sender_id=sender_id,
            timestamp=self.sim_time,        # <-- SIM TIME
            data=data,
            recipients=None
        )
        self.send_message(msg)
    
    def send_to(self, sender_id: str, recipient_id: str, msg_type: MessageType, data: Dict[str, Any]):
        """Send message to specific recipient"""
        msg = Message(
            msg_id="",
            msg_type=msg_type,
            sender_id=sender_id,
            timestamp=self.sim_time,        # <-- SIM TIME
            data=data,
            recipients={recipient_id}
        )
        self.send_message(msg)
    
    def process_communications(self, current_time: float):
        """
        Process all pending communications
        Called once per simulation step
        """
        # Process each agent's outbox
        for sender_id, messages in self.outbox.items():
            if sender_id not in self.agent_positions:
                continue
                
            sender_pos = self.agent_positions[sender_id]
            
            for msg in messages:
                # Check TTL
                if current_time - msg.timestamp > msg.ttl:
                    self._log(f"Message expired: {msg.msg_id}")
                    continue
                
                # Determine recipients
                if msg.recipients:
                    recipients = msg.recipients
                else:
                    # Broadcast - all agents except sender
                    recipients = set(self.agent_positions.keys()) - {sender_id}
                
                # Deliver to each recipient
                for recipient_id in recipients:
                    if recipient_id not in self.agent_positions:
                        continue
                    
                    # Check range
                    recipient_pos = self.agent_positions[recipient_id]
                    distance = np.linalg.norm(sender_pos - recipient_pos)
                    
                    if distance > self.comm_range:
                        self._log(f"Out of range: {sender_id} -> {recipient_id} ({distance:.0f}m)")
                        continue
                    
                    # Simulate packet loss
                    if np.random.random() < self.packet_loss_rate:
                        self._log(f"Packet lost: {msg.msg_id}")
                        continue
                    
                    # Calculate delay based on distance
                    delay = self.base_delay + (distance / self.comm_range) * 0.1
                    delivery_time = msg.timestamp + delay
                    
                    # Add to recipient's inbox with delivery time
                    self.inbox[recipient_id].append((delivery_time, msg))
        
        # Clear outboxes
        self.outbox.clear()
    
    def get_messages(self, agent_id: str, current_time: float) -> List[Message]:
        """Get all messages that have arrived for an agent"""
        if agent_id not in self.inbox:
            return []
        
        delivered = []
        remaining = deque()
        
        # Check each message in inbox
        while self.inbox[agent_id]:
            delivery_time, msg = self.inbox[agent_id].popleft()
            
            if delivery_time <= current_time:
                delivered.append(msg)
                self._log(f"Delivered: {msg.msg_type.value} to {agent_id}")
            else:
                # Not ready yet, keep in queue
                remaining.append((delivery_time, msg))
        
        # Put back messages not yet delivered
        self.inbox[agent_id] = remaining
        
        return delivered
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        total_sent = len(self.message_history)
        total_delivered = sum(
            sum(1 for _ in inbox) 
            for inbox in self.inbox.values()
        )
        
        msg_type_counts = defaultdict(int)
        for msg in self.message_history:
            msg_type_counts[msg.msg_type.value] += 1
        
        return {
            "total_messages_sent": total_sent,
            "messages_pending_delivery": total_delivered,
            "message_types": dict(msg_type_counts),
            "agents_connected": len(self.agent_positions)
        }
    
    def save_message_log(self, filename: str = "eagle_comms.json"):
        """Save message history to file"""
        log_data = {
            "stats": self.get_network_stats(),
            "messages": [msg.to_dict() for msg in self.message_history[-1000:]]  # Last 1000
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self._log(f"Saved message log to {filename}")
    
    def _log(self, message: str):
        """Internal logging"""
        if self.logger:
            self.logger.info(message)


# Testing functions
if __name__ == "__main__":
    network = CommunicationNetwork()

    # Register agents
    network.register_agent("UAV1", np.array([0, 0, 400]))
    network.register_agent("UAV2", np.array([1000, 1000, 400]))
    network.register_agent("UAV3", np.array([2000, 2000, 400]))

    # t0
    t0 = 0.0
    network.set_sim_time(t0)

    # Broadcast + unicast at t0
    network.broadcast(
        "UAV1", MessageType.THERMAL_DISCOVERED,
        {"thermal_id": "th_100_200", "position": [100, 200], "strength": 3.5, "radius": 100}
    )
    network.send_to(
        "UAV2", "UAV1", MessageType.EVENT_BID,
        {"event_id": "evt_001", "bid_cost": 1500.0, "tier": "HEALTHY_NOMAD"}
    )

    # Schedule deliveries at t0
    network.process_communications(t0)

    # Advance sim time past the base delay
    t1 = 0.25
    network.set_sim_time(t1)

    # Pull messages that have arrived by t1
    for agent_id in ["UAV1", "UAV2", "UAV3"]:
        messages = network.get_messages(agent_id, t1)
        print(f"\n{agent_id} received {len(messages)} messages:")
        for msg in messages:
            print(f"  - {msg.msg_type.value} from {msg.sender_id}")

    print("\nNetwork Stats:", network.get_network_stats())