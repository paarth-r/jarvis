import asyncio
import numpy as np
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.events import FrameEvent, SystemEvent


@pytest.mark.asyncio
async def test_subscribe_and_publish():
    bus = EventBus()
    received = []

    async def handler(event: FrameEvent):
        received.append(event)

    bus.subscribe(FrameEvent, handler)
    e = FrameEvent(camera_id="left", frame=np.zeros((800, 1280), dtype=np.uint8), timestamp=1.0)
    await bus.publish(e)

    assert len(received) == 1
    assert received[0].camera_id == "left"


@pytest.mark.asyncio
async def test_wrong_type_not_delivered():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe(FrameEvent, handler)
    await bus.publish(SystemEvent(type="test", module_id="x", payload={}))

    assert len(received) == 0


@pytest.mark.asyncio
async def test_multiple_subscribers_all_called():
    bus = EventBus()
    calls = [0, 0]

    async def h1(e): calls[0] += 1
    async def h2(e): calls[1] += 1

    bus.subscribe(FrameEvent, h1)
    bus.subscribe(FrameEvent, h2)

    e = FrameEvent(camera_id="left", frame=np.zeros((800, 1280), dtype=np.uint8), timestamp=1.0)
    await bus.publish(e)

    assert calls == [1, 1]


@pytest.mark.asyncio
async def test_publish_with_no_subscribers_is_silent():
    bus = EventBus()
    # Should not raise
    await bus.publish(FrameEvent(camera_id="x", frame=np.zeros((10, 10), dtype=np.uint8), timestamp=0.0))
