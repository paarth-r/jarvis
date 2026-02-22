"""
macOS mouse and scroll injection via Quartz Event Services.
Move, click/drag, scroll with gain and deadzone. No events when idle.
Cursor is mapped to the largest display when multiple monitors are attached.
"""
import time
from typing import Tuple

try:
    from Quartz.CoreGraphics import (
        CGEventCreateMouseEvent,
        CGEventPost,
        kCGHIDEventTap,
        kCGEventLeftMouseDown,
        kCGEventLeftMouseUp,
        kCGEventLeftMouseDragged,
        kCGEventMouseMoved,
        kCGMouseButtonLeft,
        CGEventCreateScrollWheelEvent,
        CGDisplayBounds,
        CGMainDisplayID,
    )
    _QUARTZ_AVAILABLE = True
except ImportError:
    _QUARTZ_AVAILABLE = False

try:
    from AppKit import NSScreen
    _APPKIT_AVAILABLE = True
except ImportError:
    _APPKIT_AVAILABLE = False


def _get_largest_display_bounds() -> Tuple[float, float, float, float]:
    """Return (origin_x, origin_y, width, height) of the display with largest area, in Quartz screen coordinates."""
    if not _QUARTZ_AVAILABLE:
        return (0.0, 0.0, 1920.0, 1080.0)
    display_id = CGMainDisplayID()
    if _APPKIT_AVAILABLE:
        try:
            screens = NSScreen.screens()
            if screens and len(screens) > 0:
                best_area = -1.0
                for screen in screens:
                    desc = screen.deviceDescription()
                    did = desc.get("NSScreenNumber")
                    if did is None:
                        continue
                    bounds = CGDisplayBounds(did)
                    w = float(bounds.size.width)
                    h = float(bounds.size.height)
                    area = w * h
                    if area > best_area:
                        best_area = area
                        display_id = did
        except Exception:
            pass
    bounds = CGDisplayBounds(display_id)
    ox = float(bounds.origin.x)
    oy = float(bounds.origin.y)
    w = float(bounds.size.width)
    h = float(bounds.size.height)
    return (ox, oy, w, h)


def _get_screen_size() -> Tuple[int, int]:
    """Return (width, height) of the largest display."""
    if not _QUARTZ_AVAILABLE:
        return (1920, 1080)
    _, _, w, h = _get_largest_display_bounds()
    return (int(w), int(h))


def _map_hand_to_screen(norm: float, scale: float, edge_margin: float) -> float:
    """Map hand (0-1) to screen (0-1): scale from center (e.g. 2x). Optionally clamp with edge_margin to keep cursor off edges."""
    raw = 0.5 + scale * (norm - 0.5)
    if edge_margin <= 0:
        return max(0.0, min(1.0, raw))
    return max(edge_margin, min(1.0 - edge_margin, raw))


class MouseController:
    def __init__(
        self,
        move_gain: float = 2.0,
        deadzone: float = 0.002,
        scroll_gain: float = 80.0,
        scroll_deadzone: float = 0.01,
        edge_margin: float = 0.0,
        click_halo_radius: float = 56.0,
    ):
        self._move_gain = move_gain
        self._deadzone = deadzone
        self._scroll_gain = scroll_gain
        self._scroll_deadzone = scroll_deadzone
        self._edge_margin = edge_margin
        self._click_halo_radius = click_halo_radius
        self._mouse_down = False
        self._last_emit_ts: float = 0.0
        self._min_emit_interval = 1 / 120.0  # don't spam above 120 Hz

    def _get_bounds(self) -> Tuple[float, float, float, float]:
        """Largest display: (origin_x, origin_y, width, height)."""
        return _get_largest_display_bounds()

    def _clip_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        ox, oy, w, h = self._get_bounds()
        return (
            max(ox, min(x, ox + w - 1)),
            max(oy, min(y, oy + h - 1)),
        )

    def _norm_to_screen(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """Map normalized hand position (0-1) to screen coords on the largest display."""
        sx = _map_hand_to_screen(norm_x, self._move_gain, self._edge_margin)
        sy = _map_hand_to_screen(norm_y, self._move_gain, self._edge_margin)
        ox, oy, w, h = self._get_bounds()
        return (ox + sx * w, oy + sy * h)

    def _apply_deadzone(self, dx: float, dy: float) -> Tuple[float, float]:
        if abs(dx) < self._deadzone:
            dx = 0.0
        if abs(dy) < self._deadzone:
            dy = 0.0
        return (dx, dy)

    def move(self, norm_x: float, norm_y: float) -> None:
        """Move mouse to normalized (0-1) position; (0,0) = top-left. Uses 2x scale from center."""
        if not _QUARTZ_AVAILABLE:
            return
        now = time.perf_counter()
        if now - self._last_emit_ts < self._min_emit_interval:
            return
        x, y = self._norm_to_screen(norm_x, norm_y)
        x, y = self._clip_to_screen(x, y)
        event_type = kCGEventLeftMouseDragged if self._mouse_down else kCGEventMouseMoved
        event = CGEventCreateMouseEvent(None, event_type, (x, y), kCGMouseButtonLeft)
        CGEventPost(kCGHIDEventTap, event)
        self._last_emit_ts = now

    def mouse_down(self, norm_x: float, norm_y: float) -> None:
        if not _QUARTZ_AVAILABLE:
            return
        self._mouse_down = True
        x, y = self._norm_to_screen(norm_x, norm_y)
        x, y = self._clip_to_screen(x, y)
        event = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
        CGEventPost(kCGHIDEventTap, event)

    def mouse_up(self, norm_x: float, norm_y: float) -> None:
        if not _QUARTZ_AVAILABLE:
            return
        self._mouse_down = False
        x, y = self._norm_to_screen(norm_x, norm_y)
        x, y = self._clip_to_screen(x, y)
        event = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
        CGEventPost(kCGHIDEventTap, event)

    def click(self, norm_x: float, norm_y: float) -> None:
        """Perform a normal left click at the given normalized position."""
        x, y = self._norm_to_screen(norm_x, norm_y)
        x, y = self._clip_to_screen(x, y)
        if not _QUARTZ_AVAILABLE:
            return
        print(f"[control] click at ({x:.0f}, {y:.0f})", flush=True)
        event = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
        CGEventPost(kCGHIDEventTap, event)
        event = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
        CGEventPost(kCGHIDEventTap, event)

    def scroll(self, delta_y: float) -> None:
        """Scroll vertically. delta_y: positive = scroll down, negative = scroll up."""
        if not _QUARTZ_AVAILABLE:
            return
        if abs(delta_y) < self._scroll_deadzone:
            return
        # Fixed-point delta: scale gesture value to scroll amount
        scroll_delta = int(-delta_y * self._scroll_gain)
        if scroll_delta == 0:
            return
        event = CGEventCreateScrollWheelEvent(None, 0, 1, scroll_delta)
        CGEventPost(kCGHIDEventTap, event)

    @property
    def is_mouse_down(self) -> bool:
        return self._mouse_down


def is_available() -> bool:
    return _QUARTZ_AVAILABLE
