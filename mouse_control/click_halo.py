"""
Click halo: when clicking, if a button (or clickable) is near the intended point,
activate it via Accessibility API so the click "snaps" to the control.
Uses ctypes + macOS Accessibility framework; falls back to normal click if unavailable.
"""
import ctypes
import math
from ctypes import c_void_p, c_float, c_int, byref, POINTER
from typing import Optional, Tuple


def _log(msg: str) -> None:
    print(f"[click_halo] {msg}", flush=True)

# Load CoreFoundation and ApplicationServices (AX API is in ApplicationServices on macOS)
_cf = None
_ax = None
try:
    _cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    _ax = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
    # Set up C function pointers; if any symbol is missing, disable halo
    _ax.AXUIElementCreateSystemWide.restype = c_void_p
    _ax.AXUIElementCreateSystemWide.argtypes = []
    _ax.AXUIElementCopyElementAtPosition.argtypes = [c_void_p, c_float, c_float, POINTER(c_void_p)]
    _ax.AXUIElementCopyElementAtPosition.restype = c_int  # AXError
    _ax.AXUIElementCopyAttributeValue.argtypes = [c_void_p, c_void_p, POINTER(c_void_p)]
    _ax.AXUIElementCopyAttributeValue.restype = c_int
    _ax.AXUIElementPerformAction.argtypes = [c_void_p, c_void_p]
    _ax.AXUIElementPerformAction.restype = c_int
    _cf.CFRelease.argtypes = [c_void_p]
    _cf.CFRelease.restype = None
    _kCFAllocatorDefault = None  # NULL allocator
    _cf.CFStringCreateWithCString.argtypes = [c_void_p, ctypes.c_char_p, c_int]
    _cf.CFStringCreateWithCString.restype = c_void_p
    _kCFStringEncodingUTF8 = 0x08000100
    _cf.CFStringGetCString.argtypes = [c_void_p, ctypes.c_char_p, ctypes.c_long, c_int]
    _cf.CFStringGetCString.restype = ctypes.c_bool

    def _cfstr(s: str):
        return _cf.CFStringCreateWithCString(_kCFAllocatorDefault, s.encode("utf-8"), _kCFStringEncodingUTF8)

    def _get_cf_string_value(cf_ref) -> Optional[str]:
        if not cf_ref:
            return None
        buf = ctypes.create_string_buffer(256)
        if _cf.CFStringGetCString(cf_ref, buf, 256, _kCFStringEncodingUTF8):
            return buf.value.decode("utf-8", errors="replace")
        return None
except (OSError, AttributeError):
    _cf = None
    _ax = None

if not _ax or not _cf:
    def _cfstr(s: str):
        return None

    def _get_cf_string_value(cf_ref):
        return None


def _element_at_position(x: float, y: float) -> Optional[c_void_p]:
    """Return AXUIElementRef at screen position (x, y), or None. Caller must CFRelease result."""
    if not _ax or not _cf:
        return None
    system = _ax.AXUIElementCreateSystemWide()
    if not system:
        return None
    out = c_void_p()
    err = _ax.AXUIElementCopyElementAtPosition(system, c_float(x), c_float(y), byref(out))
    if err != 0 or not out.value:
        return None
    return out.value


def _element_role(element: c_void_p) -> Optional[str]:
    """Get AXRole of element as string."""
    if not _ax or not _cf or not element:
        return None
    role_ref = c_void_p()
    err = _ax.AXUIElementCopyAttributeValue(element, _cfstr("AXRole"), byref(role_ref))
    if err != 0 or not role_ref.value:
        return None
    role = _get_cf_string_value(role_ref.value)
    _cf.CFRelease(role_ref.value)
    return role


def _element_perform_press(element: c_void_p) -> bool:
    """Perform AXPress on element. Returns True if action was performed."""
    if not _ax or not _cf or not element:
        return False
    err = _ax.AXUIElementPerformAction(element, _cfstr("AXPress"))
    return err == 0


# Roles we consider "clickable" for the halo (AXGroup does not respond to AXPress, so we don't include it)
_CLICKABLE_ROLES = frozenset({
    "AXButton", "AXLink", "AXCheckBox", "AXRadioButton", "AXMenuItem", "AXPopUpButton",
})


def find_clickable_in_halo(
    x: float, y: float, radius_px: float = 2, num_points: int = 12
) -> Optional[c_void_p]:
    """
    Sample points in a circle around (x, y). If any has a clickable element (button, link, etc.),
    return that element. Caller must CFRelease. num_points=12 gives center + 12 on circle.
    """
    if not _ax or not _cf:
        _log("AX API not available")
        return None
    best_element = None
    best_role = None
    for i in range(num_points + 1):
        if i == 0:
            px, py = x, y
        else:
            angle = (i - 1) * (2 * math.pi / num_points)
            px = x + radius_px * math.cos(angle)
            py = y + radius_px * math.sin(angle)
        el = _element_at_position(px, py)
        if not el:
            _log(f"  point ({px:.0f},{py:.0f}) no element")
            continue
        role = _element_role(el)
        if role and role in _CLICKABLE_ROLES:
            _log(f"  point ({px:.0f},{py:.0f}) role={role} -> clickable")
            if best_element and _cf:
                _cf.CFRelease(best_element)
            best_element = el
            best_role = role
        else:
            if i <= 2:
                _log(f"  point ({px:.0f},{py:.0f}) role={role or '?'}")
            if el and _cf:
                _cf.CFRelease(el)
    return best_element


def try_click_halo(x: float, y: float, radius_px: float = 12) -> bool:
    """
    If a clickable element is within radius_px of (x,y), perform AXPress on it and return True.
    Otherwise return False (caller should perform normal click at x,y).
    """
    _log(f"try_click_halo x={x:.0f} y={y:.0f} radius={radius_px:.0f}")
    element = find_clickable_in_halo(x, y, radius_px=radius_px)
    if not element:
        _log("no clickable in halo -> fallback to normal click")
        return False
    try:
        ok = _element_perform_press(element)
        _log(f"AXPress result={ok}")
        return ok
    finally:
        if element and _cf:
            _cf.CFRelease(element)
