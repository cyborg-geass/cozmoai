"""Optional learned perception modules.

Geometry remains the source of metric measurements. Perception modules add
semantic labels and evidence when a model is available.
"""

from .room_classifier import classify_room
from .schemas import RoomSemantics

__all__ = [
    "RoomSemantics",
    "classify_room",
]
