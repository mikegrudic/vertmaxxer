"""Command-line tools: ``vertmaxxer`` finds a route, ``vertmaxxer-spurify`` adds summit side trips to one."""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from . import core
from .api import OptionError, VertmaxxerError, find_route, spurify
from .core import M_TO_FT, MI_TO_M


def _latlon(text):
    try:
        lat, lon = (float(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} isn't LAT,LON; write 41.42698,-73.96568, or quote it if it has a "
                                         'space: "41.42698, -73.96568"')
    return lat, lon


def _time_h(text):
    """H:MM or H:MM:SS, or a number of hours up to 24."""
    h = _clock(text)
    if ":" not in text and h > 24:
        raise argparse.ArgumentTypeError(f"{text} hours? Give the time as H:MM, e.g. 1:30")
    return h


def _attach_negative(argv):
    """``--start -73.9,41.4`` as ``--start=-73.9,41.4``: argparse reads a value starting with "-" as an option."""
    out, it = [], iter(argv)
    for tok in it:
        nxt = next(it, None) if tok in ("--start", "--end") else None
        if nxt is not None and re.match(r"-[\d.]", nxt):
            out.append(f"{tok}={nxt}")
        else:
            out += [tok] + ([nxt] if nxt is not None else [])
    return out


def _line_buffered():
    """Show progress as it happens even when output goes to a pipe or a file."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)


def _clock(text):
    """A:BB or A:BB:CC (or a plain number) as A + BB/60 + CC/3600: M:SS in minutes, H:MM in hours."""
    return sum(float(x) / 60 ** i for i, x in enumerate(text.split(":")))


def _fmt(x):
    """Minutes as M:SS, or hours as H:MM."""
    m = round(x * 60)
    return f"{m // 60}:{m % 60:02d}"


class _Units:
    """Miles and feet, or with ``metric`` kilometers and meters: input distances and paces to miles, output in
    the user's units."""

    def __init__(self, metric):
        self.metric = metric
        self.d, self.z = ("km", "m") if metric else ("mi", "ft")
        self._m = 1000.0 if metric else MI_TO_M  # meters in a distance unit

    def to_mi(self, x):
        """A distance given in the user's units, in miles."""
        return None if x is None else x * self._m / MI_TO_M

    def per_mi(self, minutes):
        """A pace per the user's distance unit, per mile."""
        return None if minutes is None else minutes * MI_TO_M / self._m

    def dist(self, mi):
        return mi * MI_TO_M / self._m

    def gain(self, ft):
        return ft / M_TO_FT if self.metric else ft


def _settings_path():
    config = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return Path(os.environ.get("VERTMAXXER_CONFIG") or config / "vertmaxxer" / "settings.json")


def _units_setting(a):
    """The units for this run: --metric or --imperial, which are saved for later runs, or else the saved choice
    (imperial if there's none)."""
    path = _settings_path()
    try:
        saved = json.loads(path.read_text()).get("units")
    except (OSError, ValueError, AttributeError):
        saved = None
    chosen = "metric" if a.metric else "imperial" if a.imperial else None
    if chosen and chosen != saved:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"units": chosen}))
            other = "--imperial" if chosen == "metric" else "--metric"
            print(f"Using {chosen} units from now on ({other} switches back; saved in {path})")
        except OSError:
            pass
    core.METRIC = (chosen or saved) == "metric"
    return _Units(core.METRIC)


def _add_units(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument("--metric", action="store_true",
                   help="kilometers and meters (distances, --pace and --gap per km, and all output) from now on")
    g.add_argument("--imperial", action="store_true", help="miles and feet from now on (the default)")


def _print_route(r, u=_Units(False)):
    print(f"\nShape: {r.shape} ({r.details})")
    if r.end:
        print(f"Ends at {r.end[2]} ({r.end[0]:.5f}, {r.end[1]:.5f})")
    elif not r.closed:
        print(f"Ends at {r.lat[-1]:.5f}, {r.lon[-1]:.5f}")
    print(f"Distance {u.dist(r.distance_mi):.2f} {u.d}, gain {u.gain(r.gain_ft):,.0f} {u.z} "
          f"(unsmoothed {u.gain(r.gain_raw_ft):,.0f} {u.z}), {u.gain(r.gain_ft) / u.dist(r.distance_mi):,.0f} "
          f"{u.z}/{u.d}{'; optimal' if r.proven else ''}")
    for name, length in _legs(r.legs):
        print(f"  {u.dist(length / MI_TO_M):5.2f} {u.d}  {name}")


GENERIC = set(core.TRAIL_HIGHWAYS + core.ROAD_HIGHWAYS) | {"gap in map data", "sidewalk"}  # ways with no name


def _legs(legs, min_m=32.0, min_unnamed_m=80.0):
    """The turn list with legs under ``min_m`` (or unnamed ones under ``min_unnamed_m``: lot aisles, connectors,
    mapping gaps) folded into the leg before them."""
    out = []
    for name, length in legs:
        if out and (length < (min_unnamed_m if name in GENERIC else min_m) or out[-1][0] == name):
            out[-1][1] += length
        else:
            out.append([name, length])
    return out


def _writable(p, path, what):
    """Fail now, not after the solve, if ``path`` can't be written."""
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder) or not os.access(folder, os.W_OK):
        p.error(f"can't write the {what} to {path}: {folder} isn't a writable folder")


def _write(f, path):
    try:
        f(path)
    except OSError as err:
        raise SystemExit(f"Couldn't write {path}: {err}")
    print(f"Wrote {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="vertmaxxer", description=core.__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=_latlon, action="append", required=True, metavar="LAT,LON",
                   help="where to start; repeat to let the solver choose among several")
    budget = p.add_mutually_exclusive_group(required=True)
    budget.add_argument("--distance", type=float, help="max route distance (miles, or km with --metric)")
    budget.add_argument("--time", type=_time_h, metavar="H:MM", help="max route time, with --pace or --gap")
    pace = p.add_mutually_exclusive_group()
    pace.add_argument("--pace", type=_clock, metavar="M:SS",
                      help="with --time: pace per mile, or per km with --metric (sets the distance)")
    pace.add_argument("--gap", type=_clock, metavar="M:SS",
                      help="with --time: grade-adjusted pace per mile (or km) (Strava GAP); climbs and steep descents "
                           "cost more")
    p.add_argument("--topology", choices=core.TOPOLOGIES, default="lollipop", help="route shape (default lollipop)")
    p.add_argument("--end", type=_latlon, action="append", metavar="LAT,LON",
                   help="lat,lon of a finish for point-to-point routes; repeatable")
    p.add_argument("--end-trailheads", action="store_true",
                   help="end at whichever trailhead (where a trail meets a paved road open to cars) gives the most gain")
    p.add_argument("--min-end-dist", type=float,
                   help="with --end-trailheads, ignore trailheads closer than this to the start (straight line; miles, "
                        "or km with --metric; default 1 mi)")
    p.add_argument("--any-end", action="store_true",
                   help="with --end-trailheads, also allow ends on the Mount Washington Auto Road, at its summit or "
                        "on Breakneck Road")
    p.add_argument("--marked-only", action="store_true",
                   help="skip herd paths and other informal or unmarked ways (shown as \"(unmarked)\" otherwise)")
    p.add_argument("--primary-roads", action="store_true",
                   help="allow running along primary roads (often a town's main street), not just across them")
    p.add_argument("--max-road-fraction", type=float, metavar="F",
                   help="at most this share of the route's distance on roads, e.g. a road walk between trailheads "
                        "(default 0.1; 0 for trails only). Road crossings and the start's own roads don't count")
    p.add_argument("--trailhead-roads", type=float, default=400.0, metavar="M",
                   help="roads lying within M meters of the start are walkable: its lots and access roads (default 400)")
    p.add_argument("--roads", action="store_true", help="allow roads as well as trails, any amount")
    p.add_argument("--roads-only", action="store_true",
                   help="roads only, paved or dirt: no trails, tracks, driveways or parking aisles; highways only crossed")
    p.add_argument("--paved-only", action="store_true", help="leave out roads tagged unpaved (gravel, dirt, ...)")
    p.add_argument("--road-time-frac", type=float, metavar="F",
                   help="allow roads, up to this share of the route's grade-adjusted (GAP) time, e.g. 0.1")
    p.add_argument("--ways", metavar="FILE",
                   help='JSON {"include": [OSM way ids], "exclude": [...]}: ways to allow whatever their type, or to avoid')
    p.add_argument("--closures", action="append", default=[], metavar="FILE",
                   help="JSON of closed OSM segments to avoid, besides the bundled ones; repeatable")
    p.add_argument("--minimize", action="store_true",
                   help="find the route that climbs the least, covering at least 98%% of --distance")
    p.add_argument("--min-loop", type=float, help="min length of each loop (miles, or km with --metric; default 1 mi)")
    p.add_argument("--min-loop-frac", type=float, default=0.25, help="min length of each loop, as a fraction of the route")
    p.add_argument("--max-sac", type=int, choices=range(1, 7), metavar="1-6",
                   help="exclude trails above this SAC scale grade (T1-T6)")
    p.add_argument("--dem", choices=["auto", "3dep", "terrarium"], default="auto",
                   help="elevation: USGS 3DEP (US) or AWS terrain tiles (worldwide); default: 3DEP in the US")
    p.add_argument("--smooth", type=float, default=50.0, help="elevation smoothing e-folding length (m)")
    p.add_argument("--seg-max", type=float, default=500.0, help="turnaround resolution when spurs are allowed (m)")
    p.add_argument("--time-limit", type=float,
                   help="solver time limit (s; default 120, or 600 for figure-8, dumbbell and double-lollipop)")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="solver threads")
    p.add_argument("--verbose", action="store_true", help="show the CP-SAT search log")
    p.add_argument("-o", "--output", default="vertmaxxer.gpx", help="GPX to write (default vertmaxxer.gpx)")
    p.add_argument("--plot", help="also write a map + profile figure here (needs matplotlib)")
    _add_units(p)
    a = p.parse_args(_attach_negative(sys.argv[1:] if argv is None else argv))
    _line_buffered()
    u = _units_setting(a)
    if a.max_road_fraction is not None and (a.roads or a.roads_only):
        p.error("--max-road-fraction limits roads on trail routes; --roads and --roads-only allow any amount")
    _writable(p, a.output, "route")
    if a.plot:
        _writable(p, a.plot, "plot")
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError:
            p.error('--plot needs matplotlib: pip install "vertmaxxer[plot]"')
        kinds = FigureCanvasAgg(Figure()).get_supported_filetypes()
        if os.path.splitext(a.plot)[1].lstrip(".").lower() not in kinds:
            p.error(f"--plot needs a file name ending in one of: {', '.join('.' + k for k in sorted(kinds))}")

    try:
        r = find_route(a.start, u.to_mi(a.distance), a.topology, time_h=a.time, pace=u.per_mi(a.pace),
                       gap=u.per_mi(a.gap), end=a.end, end_trailheads=a.end_trailheads,
                       min_end_dist_mi=1.0 if a.min_end_dist is None else u.to_mi(a.min_end_dist), any_end=a.any_end,
                       max_road_fraction=0.1 if a.max_road_fraction is None else a.max_road_fraction,
                       marked_only=a.marked_only, primary_roads=a.primary_roads,
                       trailhead_roads_m=a.trailhead_roads, roads=a.roads, roads_only=a.roads_only,
                       paved_only=a.paved_only, road_time_frac=a.road_time_frac, ways=a.ways, closures=a.closures,
                       minimize=a.minimize, min_loop_mi=1.0 if a.min_loop is None else u.to_mi(a.min_loop),
                       min_loop_frac=a.min_loop_frac,
                       max_sac=a.max_sac,
                       dem=a.dem, smooth_m=a.smooth, seg_max_m=a.seg_max, time_limit_s=a.time_limit,
                       workers=a.workers, verbose=a.verbose)
    except OptionError as err:
        p.error(str(err))
    except VertmaxxerError as err:
        raise SystemExit(str(err))
    _print_route(r, u)
    if len(a.start) > 1:
        print(f"Starts at {r.lat[0]:.5f}, {r.lon[0]:.5f}")
    if a.gap:
        gap = u.per_mi(a.gap)
        back = f"; run in reverse, {_fmt(r.flat_mi(True) * gap / 60)}" if r.closed else ""
        print(f"Grade-adjusted distance {u.dist(r.flat_mi()):.2f} {u.d}: {_fmt(r.flat_mi() * gap / 60)} at "
              f"{_fmt(a.gap)}/{u.d} GAP{back}")
    elif a.pace:
        print(f"{_fmt(r.distance_mi * u.per_mi(a.pace) / 60)} at {_fmt(a.pace)}/{u.d}")
    if a.topology != "any" and r.shape != a.topology:
        print(f"Warning: asked for a {a.topology}, but the best route found has another shape ({r.shape})",
              file=sys.stderr)
    _write(r.write_gpx, a.output)
    if a.plot:
        _write(lambda path: r.plot(path, u.metric), a.plot)


def spurify_main(argv=None):
    p = argparse.ArgumentParser(
        prog="vertmaxxer-spurify", formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False,
        description="Add out-and-back side trips to named summits to a route you like, keeping the route itself.\n"
                    "Side trips turn around only at peaks (or where the route already turns around).")
    p.add_argument("gpx")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--extra", type=float, help="miles (or km with --metric) that may be added")
    g.add_argument("--budget", type=float, help="total miles (or km with --metric)")
    p.add_argument("-o", "--output", help="GPX to write (default ROUTE_spurred.gpx)")
    p.add_argument("--time-limit", type=float, default=60, help="solver time limit (s, default 60)")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--match-m", type=float, default=15, help="how close the track must follow a trail (m)")
    p.add_argument("--summit-m", type=float, default=60, help="how close a trail must pass a peak (m)")
    p.add_argument("--dem", choices=["auto", "3dep", "terrarium"], default="auto",
                   help="elevation: USGS 3DEP (US) or AWS terrain tiles (worldwide); default: 3DEP in the US")
    _add_units(p)
    a = p.parse_args(argv)
    _line_buffered()
    u = _units_setting(a)
    out = a.output or os.path.splitext(a.gpx)[0] + "_spurred.gpx"
    _writable(p, out, "route")

    try:
        s = spurify(a.gpx, u.to_mi(a.extra), u.to_mi(a.budget), time_limit_s=a.time_limit, workers=a.workers, match_m=a.match_m,
                    summit_m=a.summit_m, dem=a.dem)
    except OptionError as err:
        p.error(str(err))
    except VertmaxxerError as err:
        raise SystemExit(str(err))
    _write(lambda path: s.route.write_gpx(path, os.path.splitext(os.path.basename(path))[0]), out)

    print(f"\n{'Leaves track':>12s}  {'Out and back':>12s}  {'Gain':>8s}  {u.z + '/' + u.d:>6s}  Summits")
    for t in s.side_trips:
        names = ", ".join(t.summits) or "(connector, no summit)"
        print(f"{u.dist(t.leaves_at_mi):9.2f} {u.d}  {u.dist(t.length_mi):9.2f} {u.d}  {u.gain(t.gain_ft):6,.0f} {u.z:2s}  "
              f"{u.gain(t.gain_ft) / u.dist(t.length_mi):6,.0f}  {names}")
    if not s.side_trips:
        print("  (none: no summit side trip fits the budget)")
    print(f"\nBefore: {u.gain(s.base_gain_ft):,.0f} {u.z} over {u.dist(s.base_distance_mi):.2f} {u.d}.  After: "
          f"{u.gain(s.route.gain_ft):,.0f} {u.z} over {u.dist(s.route.distance_mi):.2f} {u.d} "
          f"({u.gain(s.route.gain_ft - s.base_gain_ft):+,.0f} {u.z})"
          f"{'; optimal' if s.route.proven else ''}.")
