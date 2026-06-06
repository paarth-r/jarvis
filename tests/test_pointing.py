import numpy as np
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.pointing import (
    is_pointing, compute_ray, PointingModule,
    WRIST, INDEX_MCP, INDEX_PIP, INDEX_TIP,
    MIDDLE_PIP, MIDDLE_TIP, RING_PIP, RING_TIP, PINKY_PIP, PINKY_TIP,
)
from jarvis.events import WorldPoseEvent, PointEvent


def _pointing_hand():
    """Index extended (+Z), other fingers curled (tips near wrist)."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # index chain extends outward along +Z
    lm[INDEX_MCP] = [0, 0, 0.05]
    lm[INDEX_PIP] = [0, 0, 0.09]
    lm[INDEX_TIP] = [0, 0, 0.15]
    # curled fingers: tip closer to wrist than pip
    for pip, tip in [(MIDDLE_PIP, MIDDLE_TIP), (RING_PIP, RING_TIP), (PINKY_PIP, PINKY_TIP)]:
        lm[pip] = [0, 0, 0.08]
        lm[tip] = [0, 0, 0.03]
    return lm


def _open_hand():
    lm = np.zeros((21, 3), dtype=np.float32)
    for pip, tip in [(INDEX_PIP, INDEX_TIP), (MIDDLE_PIP, MIDDLE_TIP),
                     (RING_PIP, RING_TIP), (PINKY_PIP, PINKY_TIP)]:
        lm[pip] = [0, 0, 0.08]
        lm[tip] = [0, 0, 0.15]   # all extended
    return lm


def test_is_pointing_true_for_pointing_hand():
    assert is_pointing(_pointing_hand()) is True


def test_is_pointing_false_for_open_hand():
    assert is_pointing(_open_hand()) is False


def test_compute_ray_origin_and_unit_direction():
    lm = _pointing_hand()
    origin, direction = compute_ray(lm)
    assert np.allclose(origin, [0, 0, 0.05], atol=1e-6)
    assert np.allclose(np.linalg.norm(direction), 1.0, atol=1e-6)
    assert np.allclose(direction, [0, 0, 1.0], atol=1e-6)


@pytest.mark.asyncio
async def test_module_publishes_point_event_when_pointing():
    bus = EventBus()
    received = []

    async def on_point(e: PointEvent):
        received.append(e)

    bus.subscribe(PointEvent, on_point)
    mod = PointingModule("pointing", {})
    await mod.setup(bus)

    await bus.publish(WorldPoseEvent(landmarks_3d=_pointing_hand(), timestamp=1.0,
                                     confidence=1.0, hand_side="right"))
    assert len(received) == 1
    assert received[0].target is None
    assert np.allclose(np.linalg.norm(received[0].direction), 1.0, atol=1e-6)


@pytest.mark.asyncio
async def test_module_silent_when_not_pointing():
    bus = EventBus()
    received = []

    async def on_point(e: PointEvent):
        received.append(e)

    bus.subscribe(PointEvent, on_point)
    mod = PointingModule("pointing", {})
    await mod.setup(bus)

    await bus.publish(WorldPoseEvent(landmarks_3d=_open_hand(), timestamp=1.0,
                                     confidence=1.0, hand_side="right"))
    assert received == []
