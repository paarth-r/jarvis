"""Action bindings: map gesture names to key/action handlers."""
import time
import pyautogui
from pynput.keyboard import Controller, Key

keyboard = Controller()
MEDIA_NEXT = Key.media_next
MEDIA_PREV = Key.media_previous
MEDIA_PLAY_PAUSE = Key.media_play_pause

COOLDOWN = 1.0
_last_action_time = 0


def _raycast():
    pyautogui.hotkey("command", "space")


def _media_next():
    keyboard.press(MEDIA_NEXT)
    keyboard.release(MEDIA_NEXT)


def _media_prev():
    keyboard.press(MEDIA_PREV)
    keyboard.release(MEDIA_PREV)


def _media_play_pause():
    keyboard.press(MEDIA_PLAY_PAUSE)
    keyboard.release(MEDIA_PLAY_PAUSE)


# Bind gesture names to action callables
ACTIONS = {
    "initialize": _raycast,
    "swipe_L": _media_next,
}


def perform_action(gesture: str) -> None:
    global _last_action_time
    if time.time() - _last_action_time < COOLDOWN:
        return
    if gesture in ACTIONS:
        print(f"Action: {gesture}")
        ACTIONS[gesture]()
        _last_action_time = time.time()
