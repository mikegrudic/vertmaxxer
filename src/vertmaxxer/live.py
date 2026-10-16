"""A terminal map of the best route found so far, in braille dots, redrawn in place as the search improves it."""
import shutil
import sys
import time

import numpy as np

from .core import show_z

ROUTE, NET, START, RESET = "\x1b[1;33m", "\x1b[2m", "\x1b[1;31m", "\x1b[0m"
# Braille dot bits by (row, column) within a character cell of 4 rows x 2 columns.
_BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]])


class LiveMap:
    """Called as (edges, m, gain_m, bound_m, elapsed_s, start, final=False) with each better route: the network
    dim, the route bright, the start as a dot. Redraws at most every ``every`` seconds, and always the final one."""

    def __init__(self, stream=None, width=None, height=None, every=0.25):
        self.out = stream or sys.stdout
        cols, rows = shutil.get_terminal_size((100, 40))
        self.width = width or max(20, min(cols - 1, 120))
        self.height = height or max(8, rows - 6)
        self.every, self.drawn, self.lines, self.edges = every, 0.0, 0, None

    def _layout(self, edges):
        """Each edge's dots on a grid fitted to the network (braille dots are about square)."""
        self.lat0 = np.cos(np.radians(np.mean([e["lat"][0] for e in edges])))
        xy = [np.c_[np.asarray(e["lon"]) * self.lat0, np.asarray(e["lat"])] for e in edges]
        pts = np.vstack(xy)
        self.lo = pts.min(0)
        span = np.maximum(pts.max(0) - self.lo, 1e-9)
        self.s = min((2 * self.width - 1) / span[0], (4 * self.height - 1) / span[1])
        self.W, self.H = int(span[0] * self.s) // 2 + 1, int(span[1] * self.s) // 4 + 1
        self.pix = [self._raster(p) for p in xy]
        self.net = self._cells(range(len(edges)))
        self.edges = edges

    def _dot(self, p):
        """(row, column) dots of points (lon * cos lat, lat)."""
        x, y = ((np.atleast_2d(p) - self.lo) * self.s).T
        return np.clip(np.round(4 * self.H - 1 - y), 0, 4 * self.H - 1), np.clip(np.round(x), 0, 2 * self.W - 1)

    def _raster(self, p):
        r, c = self._dot(p)
        out = []
        for i in range(len(r) - 1):  # every dot along each segment, so lines stay unbroken
            n = int(max(abs(r[i + 1] - r[i]), abs(c[i + 1] - c[i]))) + 1
            out.append(np.c_[np.linspace(r[i], r[i + 1], n + 1), np.linspace(c[i], c[i + 1], n + 1)])
        return np.unique(np.round(np.vstack(out)).astype(int), axis=0) if out else np.c_[r, c].astype(int)

    def _cells(self, ids):
        g = np.zeros((self.H, self.W), int)
        for k in ids:
            d = self.pix[k]
            np.bitwise_or.at(g, (d[:, 0] // 4, d[:, 1] // 2), _BITS[d[:, 0] % 4, d[:, 1] % 2])
        return g

    def __call__(self, edges, m, gain, bound, elapsed, start=None, final=False):
        now = time.time()
        if not final and now - self.drawn < self.every:
            return
        if edges is not self.edges:
            self._layout(edges)
        route = self._cells(np.flatnonzero(np.asarray(m) > 0))
        here = None
        for e in edges:
            if start in (e["u"], e["v"]):
                i = 0 if e["u"] == start else -1
                r, c = self._dot([e["lon"][i] * self.lat0, e["lat"][i]])
                here = (int(r[0]) // 4, int(c[0]) // 2)
                break
        lines = [f"{elapsed:6.1f} s  gain {show_z(gain)}  bound {show_z(bound)}{'  (final)' if final else ''}"]
        for r in range(self.H):
            row = []
            for c in range(self.W):
                if (r, c) == here:
                    row.append(f"{START}●{RESET}")
                elif route[r, c]:
                    row.append(f"{ROUTE}{chr(0x2800 + route[r, c])}{RESET}")
                elif self.net[r, c]:
                    row.append(f"{NET}{chr(0x2800 + self.net[r, c])}{RESET}")
                else:
                    row.append(" ")
            lines.append("".join(row).rstrip())
        up = f"\x1b[{self.lines}F" if self.lines else ""  # back to the top of the last frame
        self.out.write(up + "".join(f"{line}\x1b[K\n" for line in lines))
        self.out.flush()
        self.lines, self.drawn = len(lines), now
