"""Command-line tools: ``vertmaxxer`` finds a route, ``vertmaxxer-spurify`` adds summit side trips to one."""
import argparse
import os
import re

from . import core
from .api import VertmaxxerError, find_route, spurify
from .core import MI_TO_M


def _latlon(text):
    lat, lon = (float(x) for x in text.split(","))
    return lat, lon


def _clock(text):
    """A:BB or A:BB:CC (or a plain number) as A + BB/60 + CC/3600: M:SS in minutes, H:MM in hours."""
    return sum(float(x) / 60 ** i for i, x in enumerate(text.split(":")))


def _fmt(x):
    """Minutes as M:SS, or hours as H:MM."""
    m = round(x * 60)
    return f"{m // 60}:{m % 60:02d}"


def _print_route(r):
    print(f"\nShape: {r.shape} ({r.details})")
    if r.end:
        print(f"Ends at {r.end[2]} ({r.end[0]:.5f}, {r.end[1]:.5f})")
    print(f"Distance {r.distance_mi:.2f} mi, gain {r.gain_ft:,.0f} ft (unsmoothed {r.gain_raw_ft:,.0f} ft), "
          f"{r.gain_ft / r.distance_mi:,.0f} ft/mi{'; optimal' if r.proven else ''}")
    for name, length in r.legs:
        print(f"  {length / MI_TO_M:5.2f} mi  {name}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="vertmaxxer", description=core.__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=_latlon, action="append", required=True,
                   help="lat,lon of a trailhead; repeat to let the solver choose among several")
    budget = p.add_mutually_exclusive_group(required=True)
    budget.add_argument("--distance", type=float, help="max route distance (miles)")
    budget.add_argument("--time", type=_clock, metavar="H:MM", help="max route time, with --pace or --gap")
    pace = p.add_mutually_exclusive_group()
    pace.add_argument("--pace", type=_clock, metavar="M:SS", help="with --time: pace per mile (sets the distance)")
    pace.add_argument("--gap", type=_clock, metavar="M:SS",
                      help="with --time: grade-adjusted pace per mile (Strava GAP); climbs and steep descents cost more")
    p.add_argument("--topology", choices=core.TOPOLOGIES, default="lollipop")
    p.add_argument("--end", type=_latlon, action="append",
                   help="lat,lon of a finish for point-to-point routes; repeatable")
    p.add_argument("--end-trailheads", action="store_true",
                   help="end at whichever trailhead (where a trail meets a paved road open to cars) gives the most gain")
    p.add_argument("--min-end-dist", type=float, default=1.0,
                   help="with --end-trailheads, ignore trailheads closer than this to the start (miles, straight line)")
    p.add_argument("--any-end", action="store_true",
                   help="with --end-trailheads, also allow ends on the Mount Washington Auto Road, at its summit or "
                        "on Breakneck Road")
    p.add_argument("--max-road-fraction", type=float, default=0.1, metavar="F",
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
    p.add_argument("--min-loop", type=float, default=1.0, help="min length of each loop (miles)")
    p.add_argument("--min-loop-frac", type=float, default=0.25, help="min length of each loop, as a fraction of the route")
    p.add_argument("--max-sac", type=int, choices=range(1, 7), metavar="1-6",
                   help="exclude trails above this SAC scale grade (T1-T6)")
    p.add_argument("--dem", choices=["3dep", "terrarium"], default="3dep", help="elevation source")
    p.add_argument("--smooth", type=float, default=50.0, help="elevation smoothing e-folding length (m)")
    p.add_argument("--seg-max", type=float, default=500.0, help="turnaround resolution when spurs are allowed (m)")
    p.add_argument("--time-limit", type=float, default=120.0, help="solver time limit (s)")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count()), help="solver threads")
    p.add_argument("--verbose", action="store_true", help="show the CP-SAT search log")
    p.add_argument("-o", "--output", default="vertmaxxer.gpx", help="GPX to write (default vertmaxxer.gpx)")
    p.add_argument("--plot", help="also write a map + profile figure here (needs matplotlib)")
    a = p.parse_args(argv)

    try:
        r = find_route(a.start, a.distance, a.topology, time_h=a.time, pace=a.pace, gap=a.gap, end=a.end,
                       end_trailheads=a.end_trailheads, min_end_dist_mi=a.min_end_dist, any_end=a.any_end, max_road_fraction=a.max_road_fraction,
                       trailhead_roads_m=a.trailhead_roads, roads=a.roads, roads_only=a.roads_only,
                       paved_only=a.paved_only, road_time_frac=a.road_time_frac, ways=a.ways, closures=a.closures,
                       minimize=a.minimize, min_loop_mi=a.min_loop, min_loop_frac=a.min_loop_frac, max_sac=a.max_sac,
                       dem=a.dem, smooth_m=a.smooth, seg_max_m=a.seg_max, time_limit_s=a.time_limit,
                       workers=a.workers, verbose=a.verbose)
    except ValueError as err:
        p.error(str(err))
    except VertmaxxerError as err:
        raise SystemExit(str(err))
    _print_route(r)
    if a.gap:
        back = f"; run in reverse, {_fmt(r.flat_mi(True) * a.gap / 60)}" if r.closed else ""
        print(f"Grade-adjusted distance {r.flat_mi():.2f} mi: {_fmt(r.flat_mi() * a.gap / 60)} at {_fmt(a.gap)}/mi GAP"
              f"{back}")
    elif a.pace:
        print(f"{_fmt(r.distance_mi * a.pace / 60)} at {_fmt(a.pace)}/mi")
    if a.topology != "any" and r.shape != a.topology:
        print(f"Warning: requested {a.topology} but the route is a {r.shape}")
    r.write_gpx(a.output)
    print(f"Wrote {a.output}")
    if a.plot:
        r.plot(a.plot)
        print(f"Wrote {a.plot}")


def spurify_main(argv=None):
    p = argparse.ArgumentParser(
        prog="vertmaxxer-spurify", formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False,
        description="Add out-and-back side trips to named summits to a route you like, keeping the route itself.\n"
                    "Side trips turn around only at peaks (or where the route already turns around).")
    p.add_argument("gpx")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--extra", type=float, help="miles that may be added")
    g.add_argument("--budget", type=float, help="total miles")
    p.add_argument("-o", "--output", help="GPX to write (default ROUTE_spurred.gpx)")
    p.add_argument("--time-limit", type=float, default=60, help="solver time limit (s, default 60)")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count()))
    p.add_argument("--match-m", type=float, default=15, help="how close the track must follow a trail (m)")
    p.add_argument("--summit-m", type=float, default=60, help="how close a trail must pass a peak (m)")
    a = p.parse_args(argv)

    try:
        s = spurify(a.gpx, a.extra, a.budget, time_limit_s=a.time_limit, workers=a.workers, match_m=a.match_m,
                    summit_m=a.summit_m)
    except VertmaxxerError as err:
        raise SystemExit(str(err))
    out = a.output or re.sub(r"\.gpx$", "", a.gpx) + "_spurred.gpx"
    s.route.write_gpx(out, os.path.basename(out)[:-4])

    print(f"\n{'Leaves route':>12s}  {'Out and back':>12s}  {'Gain':>8s}  {'ft/mi':>6s}  Summits")
    for t in s.side_trips:
        names = ", ".join(t.summits) or "(connector, no summit)"
        print(f"{t.leaves_at_mi:9.2f} mi  {t.length_mi:9.2f} mi  {t.gain_ft:6,.0f} ft  {t.gain_ft / t.length_mi:6,.0f}  {names}")
    if not s.side_trips:
        print("  (none: no summit side trip fits the budget)")
    print(f"\nBefore: {s.base_gain_ft:,.0f} ft over {s.base_distance_mi:.2f} mi.  After: {s.route.gain_ft:,.0f} ft over "
          f"{s.route.distance_mi:.2f} mi ({s.route.gain_ft - s.base_gain_ft:+,.0f} ft)"
          f"{'; optimal' if s.route.proven else ''}.  Wrote {out}")
