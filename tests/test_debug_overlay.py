import pytest
from jarvis.modules.debug_overlay import DebugOverlayModule
from jarvis.runtime.event_bus import EventBus


@pytest.mark.asyncio
async def test_debug_overlay_sets_up_without_error():
    bus = EventBus()
    mod = DebugOverlayModule("debug_overlay", {"enabled": False})
    # setup should not raise even when OpenCV window won't open
    await mod.setup(bus)
