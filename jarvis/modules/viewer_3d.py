from __future__ import annotations

import multiprocessing
from typing import List, Optional, Tuple

from .hand_landmarks import HAND_CONNECTIONS

Point3D = Optional[Tuple[float, float, float]]

_NUM = 21
_BG = "#0d0d1a"
_BONE = "#00e676"
_JOINT = "#00bcd4"
_GRID = "#1e1e3a"
_LABEL = "#8888aa"
_DEFAULT_ELEV = 12.0
_DEFAULT_AZIM = -90.0


def _viewer_worker(queue: "multiprocessing.Queue") -> None:
    import matplotlib
    try:
        matplotlib.use("macosx")
    except Exception:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(7, 7), facecolor=_BG)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    try:
        fig.canvas.manager.set_window_title("Jarvis 3D Hand")
    except Exception:
        pass
    ax.view_init(elev=_DEFAULT_ELEV, azim=_DEFAULT_AZIM)
    plt.ion()
    plt.show(block=False)

    points: List[Point3D] = [None] * _NUM
    elev, azim = _DEFAULT_ELEV, _DEFAULT_AZIM

    while True:
        while True:
            try:
                item = queue.get_nowait()
                if item is None:
                    plt.close(fig)
                    return
                points = item
            except Exception:
                break
        try:
            elev, azim = ax.elev, ax.azim
        except Exception:
            pass
        _draw(ax, points, elev, azim, np)
        fig.canvas.draw_idle()
        plt.pause(0.033)


def _draw(ax, points: List[Point3D], elev: float, azim: float, np) -> None:
    ax.cla()
    ax.set_facecolor(_BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.4)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(_LABEL)
        axis.set_tick_params(colors=_LABEL, labelsize=7)
    ax.set_xlabel("X  -> right")
    ax.set_ylabel("Y  -> depth")
    ax.set_zlabel("Z  ^ up")
    ax.view_init(elev=elev, azim=azim)

    xs = [p[0] if p else None for p in points]
    ys = [p[1] if p else None for p in points]
    zs = [p[2] if p else None for p in points]
    valid = [(xs[i], ys[i], zs[i]) for i in range(_NUM) if points[i] is not None]

    if not valid:
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(0.2, 0.8)
        ax.set_zlim(-0.2, 0.2)
        return

    vx, vy, vz = zip(*valid)
    span = max(max(vx) - min(vx), max(vy) - min(vy), max(vz) - min(vz), 0.2)
    half = span / 2.0 + 0.05
    cx, cy, cz = float(np.mean(vx)), float(np.mean(vy)), float(np.mean(vz))
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)

    for i, j in HAND_CONNECTIONS:
        if points[i] is not None and points[j] is not None:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                    color=_BONE, linewidth=2.0, solid_capstyle="round")
    ax.scatter([x for x in xs if x is not None],
               [y for y in ys if y is not None],
               [z for z in zs if z is not None],
               c=_JOINT, s=28, depthshade=False, zorder=5)


class Viewer3D:
    """Interactive 3D hand viewer in a spawned subprocess. Drag to rotate."""

    def __init__(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._queue: "multiprocessing.Queue" = ctx.Queue(maxsize=2)
        self._process = ctx.Process(
            target=_viewer_worker, args=(self._queue,),
            daemon=True, name="jarvis-3d-viewer",
        )

    def start(self) -> None:
        self._process.start()

    def update(self, points: List[Point3D]) -> None:
        try:
            self._queue.put_nowait(points)
        except Exception:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(points)
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
