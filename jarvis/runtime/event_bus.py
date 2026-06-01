from __future__ import annotations
from typing import Any, Awaitable, Callable, Dict, List, Type, TypeVar

T = TypeVar("T")


class EventBus:
    """Synchronous-dispatch pub/sub. publish() awaits all subscribers in sequence."""

    def __init__(self) -> None:
        self._subscribers: Dict[type, List[Callable[[Any], Awaitable[None]]]] = {}

    def subscribe(
        self,
        event_type: Type[T],
        callback: Callable[[T], Awaitable[None]],
    ) -> None:
        self._subscribers.setdefault(event_type, []).append(callback)

    async def publish(self, event: Any) -> None:
        for callback in self._subscribers.get(type(event), []):
            await callback(event)
