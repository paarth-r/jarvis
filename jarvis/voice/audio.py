"""Microphone stream with mute window, and a non-blocking audio player."""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Literal

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
FRAME = 1280


class MicStream:
    """16 kHz mono int16 frames of 1280 samples, delivered to an asyncio queue."""

    def __init__(self, device: int | str | None = None) -> None:
        self._device = device
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.InputStream | None = None
        self._mute_depth = 0
        self._muted_until = 0.0
        self._lock = threading.Lock()

    def _enqueue(self, frame: np.ndarray) -> None:
        """Queue a frame unless muted or inside the post-mute tail."""
        with self._lock:
            if self._mute_depth > 0 or time.monotonic() < self._muted_until:
                return
        self._queue.put_nowait(frame)

    def _callback(self, indata: np.ndarray, frames: int, time_info: object,
                  status: object) -> None:
        assert self._loop is not None
        self._loop.call_soon_threadsafe(self._enqueue, indata[:, 0].copy())

    def _drain(self) -> None:
        """Discard frames queued before/while muted (stale or self-heard audio)."""
        while not self._queue.empty():
            self._queue.get_nowait()

    @contextmanager
    def mute(self, seconds_tail: float = 0.3) -> Iterator[None]:
        with self._lock:
            self._mute_depth += 1
        self._drain()
        try:
            yield
        finally:
            with self._lock:
                self._mute_depth -= 1
                self._muted_until = max(self._muted_until,
                                        time.monotonic() + seconds_tail)
            self._drain()

    async def frames(self) -> AsyncIterator[np.ndarray]:
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME,
            device=self._device, callback=self._callback,
        )
        self._stream.start()
        try:
            while True:
                yield await self._queue.get()
        finally:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class Player:
    """Plays samples without blocking the event loop."""

    def __init__(self, device: int | str | None = None) -> None:
        self._device = device

    async def play(self, samples: np.ndarray, sr: int) -> None:
        def run() -> None:
            sd.play(samples, sr, device=self._device)
            sd.wait()

        try:
            await asyncio.to_thread(run)
        except asyncio.CancelledError:
            sd.stop()
            raise

    def stop(self) -> None:
        sd.stop()


def tone(kind: Literal["wake", "working", "error"]) -> tuple[np.ndarray, int]:
    """Short synthesized sine blips; no asset files."""
    sr = 24000
    notes = {
        "wake": [(880.0, 0.08), (1320.0, 0.10)],
        "working": [(660.0, 0.12)],
        "error": [(330.0, 0.15), (220.0, 0.20)],
    }[kind]
    parts = []
    for freq, dur in notes:
        t = np.arange(int(sr * dur)) / sr
        env = np.minimum(1.0, np.minimum(t, dur - t) / 0.01)
        parts.append(np.sin(2 * np.pi * freq * t) * env * 0.3)
    return (np.concatenate(parts) * 32767).astype(np.int16), sr
