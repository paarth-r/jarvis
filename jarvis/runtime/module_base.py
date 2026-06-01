from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .event_bus import EventBus


class Module(ABC):
    def __init__(self, module_id: str, config: dict) -> None:
        self.module_id = module_id
        self.config = config
        self.bus: Optional["EventBus"] = None

    async def setup(self, bus: "EventBus") -> None:
        """Wire subscriptions here. Call super().setup(bus) first."""
        self.bus = bus

    @abstractmethod
    async def run(self) -> None:
        """Main coroutine. Cancelled via asyncio.CancelledError on shutdown."""
        ...

    async def teardown(self) -> None:
        """Release hardware, close files, etc."""
