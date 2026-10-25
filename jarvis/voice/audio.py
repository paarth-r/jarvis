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
    """Plays clips through one long-lived output stream, without blocking the loop.

    Opening and closing a stream per clip drops whatever the output device is
    still buffering (soundbars, AirPlay, Bluetooth), which clips the last word
    of every sentence. The stream stays open and is fed from a buffer instead.
    """

    def __init__(self, device: int | str | None = None, stream_factory=None) -> None:
        self._device = device
        self._factory = stream_factory or sd.OutputStream
        self._stream = None
        self._sr: int | None = None
        self._lock = threading.Lock()
        self._buf = np.zeros(0, np.float32)
        self._pos = 0
        self._drained = threading.Event()
        self._drained.set()

    def _callback(self, outdata, frames, time_info, status) -> None:  # PortAudio thread
        with self._lock:
            n = min(frames, len(self._buf) - self._pos)
            outdata[:n, 0] = self._buf[self._pos:self._pos + n]
            outdata[n:, 0] = 0.0
            self._pos += n
            if self._pos >= len(self._buf):
                self._drained.set()

    def _ensure_stream(self, sr: int) -> None:
        if self._stream is not None and self._sr == sr:
            return
        self.close()
        self._stream = self._factory(samplerate=sr, channels=1, dtype="float32",
                                     device=self._device, callback=self._callback)
        self._stream.start()
        self._sr = sr

    async def play(self, samples: np.ndarray, sr: int) -> None:
        data = np.asarray(samples)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        data = data.astype(np.float32).reshape(-1)
        self._ensure_stream(sr)
        with self._lock:
            self._buf, self._pos = data, 0
            self._drained.clear()
        try:
            await asyncio.to_thread(self._drained.wait, len(data) / sr + 5.0)
            # The device is still playing what it buffered; let it finish so the
            # mic's mute window covers the whole clip.
            await asyncio.sleep(float(getattr(self._stream, "latency", 0.0) or 0.0))
        except asyncio.CancelledError:
            self.stop()
            raise

    def stop(self) -> None:
        with self._lock:
            self._pos = len(self._buf)
            self._drained.set()

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


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
