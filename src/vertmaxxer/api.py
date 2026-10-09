"""Python API: find a route, or add summit side trips to one. The command-line tools are thin wrappers over these."""
import contextlib
import functools
import inspect
import io
import sys
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from . import core
from .core import MI_TO_M, M_TO_FT, OptionError, VertmaxxerError


@dataclass
class Route:
    """A solved route. Arrays run along the track; ``z`` is smoothed elevation (m), ``z_raw`` unsmoothed."""
    lat: np.ndarray
    lon: np.ndarray
    z: np.ndarray
    z_raw: np.ndarray
    dist: np.ndarray  # m from the start
    legs: list  # [(way name, m), ...]
    shape: str
    details: str
    proven: bool  # the solver proved no better route exists
    end: tuple = None  # (lat, lon, label) for a traverse to a trailhead
    edges: list = field(default=None, repr=False)  # the network searched, for plotting
    anchors: list = field(default=None, repr=False)

    @property
    def distance_mi(self):
        return float(self.dist[-1] / MI_TO_M)

    @property
    def gain_ft(self):
        """Gain from elevation smoothed over ~50 m (the figure every route is ranked by)."""
        return float(np.sum(np.maximum(0, np.diff(self.z))) * M_TO_FT)

    @property
    def gain_raw_ft(self):
        return float(np.sum(np.maximum(0, np.diff(self.z_raw))) * M_TO_FT)

    def flat_mi(self, reverse=False):
        """Grade-adjusted (flat-equivalent) distance in this route's direction, or the reverse (Strava's GAP model)."""
        ds = core._haversine(self.lat[:-1], self.lon[:-1], self.lat[1:], self.lon[1:])
        grade = 100 * np.diff(self.z) / np.maximum(ds, 1e-6)
        return float(np.sum(ds * core.gap_factor(-grade if reverse else grade)) / MI_TO_M)

    @property
    def closed(self):
        return core._haversine(self.lat[0], self.lon[0], self.lat[-1], self.lon[-1]) < 1.0

    @property
    def distance_km(self):
        return float(self.dist[-1] / 1000)

    @property
    def gain_m(self):
        return self.gain_ft / M_TO_FT

    def write_gpx(self, path, name=None):
        core.write_gpx(path, vars(self), name or f"{self.shape} {self.distance_mi:.1f} mi")

    def plot(self, path, metric=False):
        """Map over the searched network, and elevation profile (needs matplotlib), in ft and mi or m and km."""
        core.plot(path, self.edges, vars(self), self.anchors, metric)


def _route(edges, m, start, end, proven, min_loop, min_loop_frac, heads=None, anchors=None, access=None):
    """The solved route; with ``access`` (the real start, and the edges from it to the solver's start), that path is
    run out and back around it without counting toward the shape."""
    shape, details = core.classify(edges, m, start, end, min_loop, min_loop_frac)
    if access and sum(edges[k]["length"] for k in access[1]) >= 16:
        m = np.array(m)
        for k in access[1]:
            m[k] += 2
        details += f"; plus {core.show_d(sum(edges[k]['length'] for k in access[1]))} access path each way"
        start = end = access[0]
    r = core.assemble(edges, m, start, end)
    return Route(lat=r["lat"], lon=r["lon"], z=r["z"], z_raw=r["z_raw"], dist=r["dist"], legs=r["legs"], shape=shape,
                 details=details, proven=bool(proven), end=(heads or {}).get(end), edges=edges, anchors=anchors)


def _walk_to_trails(edges, start, max_m):
    """Ids of the road edges on the shortest walk from ``start`` to the nearest trail network with at least
    MIN_TRAIL_NET of trail, if one is within ``max_m``."""
    T = nx.Graph()
    for e in edges:
        if not e["road"]:  # a loop with one junction is a self-loop: it counts toward its network's size
            T.add_edge(e["u"], e["v"], w=e["length"])
    trails = {n for c in nx.connected_components(T) if T.subgraph(c).size(weight="w") >= core.MIN_TRAIL_NET for n in c}
    G, by = nx.Graph(), {}
    for e in edges:
        if e["u"] != e["v"] and G.get_edge_data(e["u"], e["v"], {}).get("w", np.inf) > e["length"]:
            G.add_edge(e["u"], e["v"], w=e["length"])
            by[frozenset((e["u"], e["v"]))] = e
    if start not in G or start in trails:
        return set()
    dist, path = nx.single_source_dijkstra(G, start, cutoff=max_m, weight="w")
    near = min((n for n in dist if n in trails), key=dist.get, default=None)
    if near is None:
        return set()
    return {id(by[frozenset(p)]) for p in zip(path[near], path[near][1:]) if by[frozenset(p)]["road"]}


def _nearer_street(osm, point, min_path_m=100.0):
    """Whether a street is nearer ``point`` than any trail (a start in town, say, rather than at a trailhead).
    Paths in networks shorter than ``min_path_m`` (a traffic island, a crosswalk stub) don't count as trails."""
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm["elements"] if el["type"] == "node"}
    best = {True: np.inf, False: np.inf}
    paths = nx.Graph()
    for w in osm["elements"]:
        tags = w.get("tags", {})
        if w["type"] != "way" or "highway" not in tags or not core._usable(tags, True, None) or core._not_a_start(tags):
            continue
        ll = np.array([nodes[n] for n in w["nodes"] if n in nodes])
        if not len(ll):
            continue
        if core._is_road(tags["highway"]):
            best[True] = min(best[True], float(core._haversine(*point, ll[:, 0], ll[:, 1]).min()))
        else:
            ids = [n for n in w["nodes"] if n in nodes]
            for a, b in zip(ids, ids[1:]):
                paths.add_edge(a, b, w=float(core._haversine(*nodes[a], *nodes[b])))
    for comp in nx.connected_components(paths):
        if paths.subgraph(comp).size(weight="w") >= min_path_m:
            ll = np.array([nodes[n] for n in comp])
            best[False] = min(best[False], float(core._haversine(*point, ll[:, 0], ll[:, 1]).min()))
    return best[True] <= best[False]  # a node shared by a street and a path: start on the street, right there


TIME_LIMITS = {"figure-8": 600.0, "dumbbell": 600.0, "double-lollipop": 600.0}  # s; others 120: two loops are hard
ACCESS_MAX_M = 1609.344  # how far a loop-shaped route may walk out to its loop
TURN_PENALTY_M = 9.144  # 30 ft: a turnaround must climb at least this much to be worth it


def _collapse_parallel(edges, short_m=100.0):
    """Of ways joining the same two points, all shorter than ``short_m``, only the shortest: two footways side by
    side are one way, not a mini-loop (or out-and-back) to pick up a few feet."""
    pairs = {}
    for e in edges:
        pairs.setdefault(frozenset((e["u"], e["v"])), []).append(e)
    out = []
    for es in pairs.values():
        out += [min(es, key=lambda e: e["length"])] if max(e["length"] for e in es) < short_m else es
    return out


def _tidy(edges, keep, short_m=100.0):
    """The network without clutter that routes turning back anywhere would exploit: parallel short ways, and
    dead-end road stubs shorter than ``short_m`` (alleys, lot entrances; repeatedly, so a short street that ends in
    stubs goes too)."""
    edges = _collapse_parallel(edges, short_m)
    while True:
        deg = Counter(n for e in edges if e["u"] != e["v"] for n in (e["u"], e["v"]))
        stubs = {id(e) for e in edges if e["road"] and e["length"] < short_m
                 and any(deg[n] == 1 and n not in keep for n in (e["u"], e["v"]))}
        if not stubs:
            return edges
        edges = [e for e in edges if id(e) not in stubs]


def _graph(edges, avoid=(), road_weight=1.0):
    """Simple graph of ``edges`` clear of the nodes in ``avoid``, keeping the lightest of parallel edges, with road
    meters weighted ``road_weight`` times (to keep shortest paths off roads)."""
    G = nx.Graph()
    for k, e in enumerate(edges):
        w = e["length"] + (road_weight - 1) * e.get("road_len", 0.0)
        if e["u"] != e["v"] and e["u"] not in avoid and e["v"] not in avoid \
                and G.get_edge_data(e["u"], e["v"], {}).get("w", np.inf) > w:
            G.add_edge(e["u"], e["v"], w=w, k=k)
    return G


def _loops_through(edges, node, lo, hi, avoid=(), tries=30, road_weight=1.0):
    """Simple loops through ``node``, ``lo`` to ``hi`` m long and clear of the nodes in ``avoid``, as lists of indices
    into ``edges``, climbiest first: two node-disjoint shortest paths to each of up to ``tries`` turnarounds."""
    G = _graph(edges, avoid, road_weight)
    if node not in G:
        return []
    dist, path = nx.single_source_dijkstra(G, node, cutoff=hi / 2, weight="w")
    far = sorted((n for n in dist if dist[n] >= lo / 4), key=dist.get)
    found = []
    for X in far[:: max(1, len(far) // tries)]:
        p1 = path[X]
        H = nx.restricted_view(G, p1[1:-1], [(p1[0], p1[1])] if len(p1) == 2 else [])
        try:
            _, p2 = nx.single_source_dijkstra(H, node, X, cutoff=2 * hi, weight="w")
        except nx.NetworkXNoPath:
            continue
        ks = [G[u][v]["k"] for q in (p1, p2) for u, v in zip(q, q[1:])]
        if lo <= sum(edges[k]["length"] for k in ks) <= hi:
            found.append((sum(edges[k]["var"] for k in ks), ks))
    return [ks for _, ks in sorted(found, key=lambda f: -f[0])]


def _seed(edges, start, budget, shape, min_loop, min_loop_frac, max_road_frac=None, stems=16, min_length=0.0,
          minimize=False):
    """A quick route to start the solver from, as a count per edge: for a loop, a loop through the start; for a
    lollipop, the climbiest (with ``minimize``, the flattest) of a few loops at the ends of short stems; within
    ``max_road_frac`` of road, if given, and at least ``min_length`` long. None for other shapes, or if none fits.
    Dense networks can otherwise take the solver minutes just to find a first route."""
    rw = 1.0 if max_road_frac is None else 5.0  # keep off roads where they're capped
    out_and_back = shape["loops"] in (0, None) and shape["spurs"] != 0 and shape["start"] in ("stem", None)
    if not out_and_back and (shape["loops"] != 1 or shape["spurs"] != 0 or shape["start"] not in ("loop", "stem")):
        return None

    def counts_of(loop, stem=()):
        c = np.zeros(len(edges), int)
        c[list(loop)] = 1
        c[list(stem)] = 2
        if max_road_frac is not None:
            road = sum(edges[k]["road_len"] * c[k] for k in np.flatnonzero(c))
            if road > max_road_frac * sum(edges[k]["length"] * c[k] for k in np.flatnonzero(c)):
                return None
        return c

    def best(cands):
        cands = [c for c in cands if c is not None]
        climb = lambda c: float(np.dot(c, [e["var"] for e in edges])) * (-1 if minimize else 1)
        return max(cands, key=climb, default=None)

    if out_and_back:  # also a valid start for "any": the climbiest shortest path there and back
        if minimize:
            return None
        G = _graph(edges, road_weight=rw)
        if start not in G:
            return None
        _, path = nx.single_source_dijkstra(G, start, cutoff=budget / 2 * rw, weight="w")
        cands = []
        for X, q in path.items():
            ks = [G[u][v]["k"] for u, v in zip(q, q[1:])]
            if ks and 2 * sum(edges[k]["length"] for k in ks) <= budget:
                cands.append(counts_of((), ks))
        return best(cands)
    if shape["start"] == "loop":
        lo = max(min_loop, min_length)
        return best(counts_of(ks) for ks in _loops_through(edges, start, lo, budget, road_weight=rw,
                                                          tries=200 if minimize else 30))
    G = _graph(edges, road_weight=rw)
    if start not in G:
        return None
    dist, path = nx.single_source_dijkstra(G, start, cutoff=budget / 4, weight="w")
    cands = []
    for X in sorted((n for n in dist if dist[n] > 0), key=dist.get)[:: max(1, len(dist) // stems)]:
        stem = [G[u][v]["k"] for u, v in zip(path[X], path[X][1:])]
        d = sum(edges[k]["length"] for k in stem)
        lo = max(min_loop, 2 * d * min_loop_frac / (1 - min_loop_frac) * 1.01, min_length - 2 * d)
        # A minimize seed must land in the narrow band from min_length to the budget: try many more turnarounds.
        loops = _loops_through(edges, X, lo, budget - 2 * d, avoid=set(path[X][:-1]), tries=80 if minimize else 10,
                               road_weight=rw)
        for ks in (loops[-3:] if minimize else loops[:3]):
            cands.append(counts_of(ks, stem))
    return best(cands)


# Rough boxes (lat, lat, lon, lon) around where 3DEP has data: the lower 48, Alaska, Hawaii, Puerto Rico and the
# Virgin Islands, Guam. They take in some of Canada and Mexico too, where 3DEP may or may not have data.
US = [(24.3, 49.5, -125.0, -66.8), (51.0, 71.5, -180.0, -129.9), (18.8, 22.4, -160.3, -154.7),
      (17.6, 18.6, -67.4, -64.5), (13.2, 13.7, 144.6, 145.0)]


def _dem_for(points, dem):
    """``dem``, or with "auto", 3DEP for points in the US and AWS terrain tiles elsewhere."""
    if dem != "auto":
        return dem
    if all(any(a <= la <= b and c <= lo <= d for a, b, c, d in US) for la, lo in points):
        return "3dep"
    print("Outside the US: elevation from AWS terrain tiles (coarser than 3DEP in most places)")
    return "terrarium"


def _points(p, dem="3dep"):
    """One (lat, lon), or "lat,lon", or a list of them, as a list of checked (lat, lon)."""
    if isinstance(p, str):
        p = p.split(",")
    elif isinstance(p, (list, tuple)) and p and all(isinstance(q, str) for q in p) and "," in p[0]:
        p = [q.split(",") for q in p]
    try:
        pts = [tuple(map(float, p))] if np.ndim(p) == 1 else [tuple(map(float, q)) for q in p]
    except (TypeError, ValueError):
        raise OptionError(f"{p!r} isn't a (lat, lon) point or a list of them")
    for la, lo in pts:
        if not (math.isfinite(la) and math.isfinite(lo) and -90 <= la <= 90 and -180 <= lo <= 180):
            raise OptionError(f"{la}, {lo} isn't a latitude, longitude")
        in_us = lambda la, lo: any(a <= la <= b and c <= lo <= d for a, b, c, d in US)
        if not in_us(la, lo) and in_us(lo, la):
            raise OptionError(f"{la}, {lo} is far from anywhere; did you swap latitude and longitude?")
        if dem == "3dep" and not in_us(la, lo):
            raise OptionError(f"{la}, {lo} is outside the US, where 3DEP elevation is available: check the order "
                              "(latitude first), or use the terrarium elevation source (--dem terrarium)")
    return pts


def _quietly(f):
    """Run ``f`` with its progress output discarded when called with ``quiet=True``."""
    @functools.wraps(f)
    def g(*a, quiet=False, **k):
        if not quiet:
            return f(*a, **k)
        with contextlib.redirect_stdout(io.StringIO()):
            return f(*a, **k)
    sig = inspect.signature(f)
    g.__signature__ = sig.replace(parameters=[*sig.parameters.values(),
                                              inspect.Parameter("quiet", inspect.Parameter.KEYWORD_ONLY, default=False)])
    return g


@_quietly
def find_route(start, distance_mi=None, topology="lollipop", *, time_h=None, pace=None, gap=None, end=None,
               end_trailheads=False, min_end_dist_mi=1.0, max_road_fraction=0.1, trailhead_roads_m=400.0, roads=False, roads_only=False, paved_only=False,
               road_time_frac=None, ways=None, closures=(), any_end=False, minimize=False, min_loop_mi=1.0,
               min_loop_frac=0.25, max_sac=None, dem="auto", smooth_m=50.0, seg_max_m=500.0, time_limit_s=None,
               workers=8, verbose=False, marked_only=False, primary_roads=False):
    """The route with the most climbing (or with ``minimize``, the least) from ``start``.

    ``start`` is (lat, lon), or a list of them to let the solver pick. ``distance_mi`` is the most the route may
    run; or give ``time_h`` with ``pace`` (minutes per mile, which just sets the distance) or ``gap`` (grade-adjusted
    minutes per mile: each stretch then costs its Strava GAP-equivalent flat distance; a stretch run once counts the
    average of its two directions). ``topology`` is one of ``core.TOPOLOGIES``; ``traverse`` needs ``end`` (a point
    or list) or ``end_trailheads``. Roads: by default up to ``max_road_fraction`` of the distance, besides road crossings and
    the roads within ``trailhead_roads_m`` of the start; ``roads`` allows any amount; ``roads_only`` uses streets
    alone. ``ways`` is a dict (or JSON path) of OSM way ids, {"include": [...], "exclude": [...]}: included ways
    are used whatever their tags (e.g. a sidewalk), excluded ones never. ``closures`` are JSON files (a path or a
    list) of closed segments, besides the bundled ones. ``max_sac`` (1-6) skips harder trails, and
    ``marked_only`` herd paths and other informal or unmarked ways. Primary roads are only crossed unless
    ``primary_roads`` (trunk roads always). ``quiet`` silences progress output.

    Raises OptionError (a ValueError) for invalid options, and VertmaxxerError if no route fits or a data source
    fails.
    """
    if time_limit_s is None:
        time_limit_s = TIME_LIMITS.get(topology, 120.0)
    for name, x in dict(distance_mi=distance_mi, time_h=time_h, pace=pace, gap=gap, time_limit_s=time_limit_s,
                        seg_max_m=seg_max_m, workers=workers).items():
        if x is not None and not (x > 0 and math.isfinite(x)):
            raise OptionError(f"{name} must be a positive number")
    for name, x, hi in (("max_road_fraction", max_road_fraction, 1), ("min_loop_frac", min_loop_frac, 0.99),
                        ("min_loop_mi", min_loop_mi, np.inf), ("trailhead_roads_m", trailhead_roads_m, np.inf)):
        if not 0 <= x <= hi:
            raise OptionError(f"{name} must be between 0 and {hi}")
    budget_flat = None  # grade-adjusted (flat-equivalent) m, with a time at a GAP
    if time_h is not None:
        if distance_mi is not None:
            raise OptionError("give a distance or a time, not both")
        if (pace is None) == (gap is None):
            raise OptionError("with a time, give one of pace or gap (minutes per mile)")
        if pace is not None:
            distance_mi = time_h * 60 / pace
        elif minimize:
            raise OptionError("minimize needs a distance, or a time with a pace")
        else:
            budget_flat = time_h * 60 / gap * MI_TO_M
            g = np.linspace(0, 50, 501)
            distance_mi = budget_flat / MI_TO_M / np.min((core.gap_factor(g) + core.gap_factor(-g)) / 2)
    elif distance_mi is None:
        raise OptionError("give distance_mi, or time_h with pace or gap")
    if topology not in core.TOPOLOGIES:
        raise OptionError(f"unknown topology {topology!r}; one of {', '.join(core.TOPOLOGIES)}")
    shape = core.TOPOLOGIES[topology]
    starts_ll = _points(start, dem)
    ends_ll = _points(end, dem) if end is not None else []
    dem = _dem_for(starts_ll + ends_ll, dem)
    if any(core._haversine(*p, *q) < 200 for p in ends_ll for q in starts_ll):
        raise OptionError("the end is at the start: for a route back to the start, use a loop or another closed shape")
    if shape["loops"] and distance_mi < min_loop_mi:
        raise OptionError(f"a {topology} needs loops of at least {min_loop_mi:g} mi (min_loop_mi), more than the "
                          f"{distance_mi:.2f} mi allowed")
    p2p = bool(ends_ll) or end_trailheads
    if ends_ll and end_trailheads:
        raise OptionError("give either end or end_trailheads")
    if p2p and topology not in ("traverse", "any"):
        raise OptionError(f"{topology} routes return to the start; use traverse or any for point-to-point")
    if topology == "traverse" and not p2p:
        raise OptionError("traverse needs end or end_trailheads")
    if roads_only and end_trailheads:
        raise OptionError("end_trailheads finds trail ends; with roads_only give end")
    if isinstance(closures, (str, os.PathLike)):
        closures = [closures]
    for f in list(closures) + ([ways] if isinstance(ways, (str, os.PathLike)) else []):
        if not os.path.isfile(f):
            raise OptionError(f"no such file: {f}")
    if isinstance(ways, (str, os.PathLike)):
        try:
            ways = json.load(open(ways))
        except ValueError as err:
            raise OptionError(f"{ways} isn't valid JSON: {err}")
    ways = ways or {}
    if not isinstance(ways, dict):
        raise OptionError('ways must be {"include": [OSM way ids], "exclude": [...]}')
    for f in closures:
        try:
            core.load_closures([f])
        except (ValueError, KeyError, TypeError) as err:
            raise OptionError(f'{f} isn\'t a closures file ({{"closed_segments": [[node, node], ...]}}): {err!r}')
    include, exclude = set(ways.get("include", ())), set(ways.get("exclude", ()))

    budget = distance_mi * MI_TO_M
    anchors = starts_ll + ends_ll
    max_sac = None if max_sac is None else max_sac - 1
    any_roads = roads or roads_only or road_time_frac is not None
    closed = core.load_closures(sorted(core.CLOSURES_DIR.glob("*.json")) + list(closures))
    heads = {}
    if end_trailheads:  # the route may end anywhere within the full budget of the start
        osm = core.fetch_osm(starts_ll, budget, True, include)
        heads = core.road_trailheads(osm, starts_ll, budget, roads, max_sac, closed)
        heads = {n: h for n, h in heads.items()
                 if min(core._haversine(h[0], h[1], la, lo) for la, lo in starts_ll) >= min_end_dist_mi * MI_TO_M
                 and (any_end or (h[2].split(" at ")[-1] not in core.EXCLUDED_END_ROADS
                                  and all(core._haversine(h[0], h[1], la, lo) > r for la, lo, r in core.EXCLUDED_ENDS)))}
        print(f"{len(heads)} trailheads at least {core.show_d(min_end_dist_mi * MI_TO_M)} from the start")
        if not heads:
            raise VertmaxxerError(f"No trailhead within reach is at least {min_end_dist_mi:g} mi from the start; "
                                  "lower --min-end-dist or raise the distance")
    else:
        osm = core.fetch_osm(anchors, budget / 2, True, include)
    if exclude:
        osm = dict(osm, elements=[el for el in osm["elements"] if not (el["type"] == "way" and el["id"] in exclude)])
    # A start nearer a street than a trail starts on the street; the walk from it to the trails is a road walk.
    street = (list(range(len(anchors))) if roads_only
              else [k for k, p in enumerate(anchors) if _nearer_street(osm, p)])
    missing = include - {el["id"] for el in osm["elements"] if el["type"] == "way"}
    if missing:
        print(f"Warning: {len(missing)} included ways aren't in OpenStreetMap: {', '.join(map(str, sorted(missing)))}",
              file=sys.stderr)
    edges, anchor_ids = core.build_graph(osm, True, max_sac, anchors, extra_ids=heads, closed=closed, snap_roads=street,
                                         include=include, marked_only=marked_only)
    if primary_roads:
        edges = [dict(e, major=False) if e.get("highway") in ("primary", "primary_link") else e for e in edges]
    blocked = any(e.get("highway") in ("primary", "primary_link") and e.get("major") for e in edges)
    if any_roads:  # major roads only where they meet others
        edges = [e for e in edges if not e.get("major")]
    else:  # trails, and road walks between them; crossings and the start's own roads are free
        free = {id(e) for e in core.road_connectors(edges, core.CROSSING_M) if e["road"]}
        free |= core.trailhead_roads(edges, anchor_ids[: len(starts_ll)], trailhead_roads_m)
        minor = [e for e in edges if not e.get("major")]
        walks = {id(e) for e in core.road_connectors(minor, max_road_fraction * budget)}
        walks |= core.trailhead_roads(edges, [anchor_ids[k] for k in street], max_road_fraction * budget)
        free |= core.trailhead_roads(edges, anchor_ids[len(starts_ll):], trailhead_roads_m)  # an end's lot, too
        for k in street:  # from a start in town, the walk to the nearest trails (it still counts as distance)
            free |= _walk_to_trails(minor, anchor_ids[k], budget / 2)
        edges = [dict(e, roadpt=np.zeros_like(e["roadpt"])) if id(e) in free else e
                 for e in edges if id(e) in free | walks]
    if paved_only:
        edges = [e for e in edges if not (e["road"] and e.get("surface") in core.UNPAVED)]
    if roads_only:
        edges = core.road_network(edges, include)
    if road_time_frac is not None:
        edges = core.road_connectors(edges, road_time_frac * budget / core.ROAD_GAP_MIN)
    starts = anchor_ids[: len(starts_ll)]
    ends = list(heads) if end_trailheads else anchor_ids[len(starts_ll) :] or None
    # Routes that turn around mid-trail may run any edge partway; it's split at seg_max_m below.
    edges = core.contract(core.prune(edges, budget, starts, ends, partial=shape["spurs"] != 0),
                          set(anchor_ids) | set(heads))
    core.add_elevation(edges, smooth_m, dem)
    edges = core.prune(core.subdivide(edges, None if shape["spurs"] == 0 else seg_max_m), budget, starts, ends)
    if budget_flat is not None:
        for e in edges:
            e["cost"] = sum(e["gap"]) / 2
    if shape["spurs"] != 0:
        edges = _tidy(edges, set(starts) | set(ends or ()))

    # A start off any loop (a trailhead up an approach trail, a street stub) starts loop shapes at the nearest loop,
    # the path there run out and back on top of the route and charged to the budget.
    access, costs, access_road, access_gain = {}, None, None, None
    if shape["start"] == "loop":
        for st in starts:
            spur, at = core.access_spur(edges, st, ACCESS_MAX_M)
            cost = sum(sum(edges[k]["gap"]) if budget_flat else 2 * edges[k]["length"] for k in spur)
            if at not in access or cost < access[at][2]:
                access[at] = (st, spur, cost)
        starts, costs = list(access), [c for _, _, c in access.values()]
        access_road = [2 * sum(edges[k]["road_len"] for k in spur) for _, spur, _ in access.values()]
        access_gain = [sum(edges[k]["var"] for k in spur) for _, spur, _ in access.values()]
    hint = None if budget_flat or ends else _seed(
        edges, starts[0], budget - (costs[0] if costs else 0.0), shape, min_loop_mi * MI_TO_M, min_loop_frac,
        None if any_roads else max_road_fraction, min_length=0.98 * budget if minimize else 0.0, minimize=minimize)
    floor = 0.98 * budget if minimize else 0.0
    common = dict(min_loop_frac=min_loop_frac, road_time_frac=road_time_frac,
                  max_road_frac=None if any_roads else max_road_fraction,
                  turn_penalty=TURN_PENALTY_M if shape["spurs"] != 0 else 0.0, start_cost=costs,
                  start_road=access_road, start_gain=access_gain)
    budget_m = budget_flat or budget
    if minimize and hint is None:
        # A route long enough to minimize from is hard to find directly: the length has to land in a narrow band.
        # The hilliest route tends to fill the budget, so find one first and minimize from it, its length (if
        # under 98% of the budget) as the floor.
        print("Finding a long route to start from...")
        first = min(60.0, time_limit_s / 3)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                m0, s0, _, _ = core.solve(edges, starts, ends, budget_m, shape, min_loop_mi * MI_TO_M, first, workers,
                                          False, hint=_seed(edges, starts[0], budget - (costs[0] if costs else 0.0),
                                                            shape, min_loop_mi * MI_TO_M, min_loop_frac,
                                                            common["max_road_frac"]), **common)
            length = sum(e["length"] * c for e, c in zip(edges, m0)) + (costs[starts.index(s0)] if costs else 0.0)
            if length < floor:
                floor = length
                print(f"Minimizing over routes of at least {core.show_d(length)}, the longest found quickly")
            hint, time_limit_s = m0, max(time_limit_s - first, 10.0)
        except VertmaxxerError:
            pass
    try:
        m, s, t, proven = core.solve(edges, starts, ends, budget_m, shape, min_loop_mi * MI_TO_M, time_limit_s, workers,
                                     verbose, minimize=minimize, min_length=floor, hint=hint, **common)
    except VertmaxxerError as err:
        if roads_only and blocked and not primary_roads:
            raise VertmaxxerError(f"{err}. Primary roads near the start (often a town's main street) can only be "
                                  "crossed, which may cut the streets apart: allow running along them with "
                                  "--primary-roads") from None
        raise
    return _route(edges, m, s, t, proven, min_loop_mi * MI_TO_M, min_loop_frac, heads, anchors, access.get(s))


# ---------------------------------------------------------------- spurify

@dataclass
class SideTrip:
    leaves_at_mi: float  # where it leaves the original route
    length_mi: float  # out and back
    gain_ft: float
    summits: list


@dataclass
class Spurred:
    route: Route
    base_gain_ft: float
    base_distance_mi: float
    side_trips: list


def read_gpx(path):
    """Track points of a GPX file (or its route points, if it has no track), as an (n, 2) array of lat, lon."""
    try:
        root = ET.parse(path).getroot()
    except FileNotFoundError:
        raise VertmaxxerError(f"No such file: {path}")
    except (ET.ParseError, UnicodeDecodeError, IsADirectoryError) as err:
        raise VertmaxxerError(f"{path} isn't a GPX file: {err}")
    pts = {tag: [(float(e.get("lat")), float(e.get("lon"))) for e in root.iter() if e.tag.split("}")[-1] == tag
                 and e.get("lat") is not None and e.get("lon") is not None] for tag in ("trkpt", "rtept")}
    return np.array(pts["trkpt"] or pts["rtept"], float).reshape(-1, 2)


def _along(pts):
    return np.r_[0, np.cumsum(core._haversine(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1]))]


def _match(edges, pts, tol, start, end):
    """Times the track runs each edge: the junctions it passes within ``tol`` (m), in order, from ``start`` to
    ``end``, each pair joined by whichever of the few shortest trails between them best matches the track's length
    there (so the long way round a loop isn't swapped for the short one); the result is one continuous walk."""
    lat0 = np.radians(pts[:, 0].mean())
    G = nx.MultiGraph()
    xy = {}
    for k, e in enumerate(edges):
        G.add_edge(e["u"], e["v"], key=k, w=e["length"])
        for n, i in ((e["u"], 0), (e["v"], -1)):
            xy[n] = (e["lon"][i] * 111320 * np.cos(lat0), e["lat"][i] * 110540)
    nodes = list(xy)
    nxy = np.array([xy[n] for n in nodes])
    d = _along(pts)
    s = np.arange(0, d[-1], 5.0)
    trk = np.c_[np.interp(s, d, pts[:, 1]) * 111320 * np.cos(lat0), np.interp(s, d, pts[:, 0]) * 110540]
    seq, at = [start], [0.0]
    for p, si in zip(trk, s):
        dd = np.hypot(*(nxy - p).T)
        i = int(dd.argmin())
        if dd[i] <= tol and seq[-1] != nodes[i]:
            seq.append(nodes[i])
            at.append(si)
    if seq[-1] != end:
        seq.append(end)
        at.append(d[-1])
    H = nx.Graph()  # shortest parallel edge only
    for u, v, w in G.edges(data="w"):
        if u != v and (not H.has_edge(u, v) or H[u][v]["w"] > w):
            H.add_edge(u, v, w=w)
    counts = np.zeros(len(edges), int)
    for a, b, ds in zip(seq, seq[1:], np.diff(at)):
        paths = [q for _, q in zip(range(4), nx.shortest_simple_paths(H, a, b, weight="w"))]
        path = min(paths, key=lambda q: abs(nx.path_weight(H, q, "w") - ds))
        for u, v in zip(path, path[1:]):
            counts[min(G[u][v], key=lambda k: G[u][v][k]["w"])] += 1
    if (counts > 2).any():
        print(f"Warning: the route runs {(counts > 2).sum()} trail sections more than twice; counted as twice.",
              file=sys.stderr)
    return np.minimum(counts, 2)


def _runs_backward(route, pts):
    """Whether a closed ``route`` goes round the other way from the track ``pts``: where it passes the track's
    quarter-way point, as a share of its length, is nearer 3/4 than 1/4."""
    q = pts[np.searchsorted(_along(pts), _along(pts)[-1] / 4)]
    i = int(np.argmin(core._haversine(*q, route.lat, route.lon)))
    f = route.dist[i] / route.dist[-1]
    return abs(f - 0.75) < abs(f - 0.25)


def _reversed_route(r):
    rev = lambda x: np.asarray(x)[::-1]
    return Route(lat=rev(r.lat), lon=rev(r.lon), z=rev(r.z), z_raw=rev(r.z_raw), dist=r.dist[-1] - rev(r.dist),
                 legs=r.legs[::-1], shape=r.shape, details=r.details, proven=r.proven, end=r.end, edges=r.edges,
                 anchors=r.anchors)


@_quietly
def spurify(track, extra_mi=None, budget_mi=None, *, time_limit_s=60.0, workers=8, match_m=15.0, summit_m=60.0,
            dem="auto"):
    """Add out-and-back side trips to named summits to a route, keeping the route itself.

    ``track`` is a GPX path or an (n, 2) array of lat, lon. Give ``extra_mi`` (miles that may be added) or
    ``budget_mi`` (total miles). Side trips turn around only at peaks within ``summit_m`` of a trail (or where the
    route already turns around); ``match_m`` is how closely the track must follow the trails. The extra miles
    count from the length of trail the track matches.
    """
    if (extra_mi is None) == (budget_mi is None):
        raise OptionError("give one of extra_mi or budget_mi")
    for name, x in dict(extra_mi=extra_mi, budget_mi=budget_mi).items():
        if x is not None and not (x >= 0 and math.isfinite(x)):
            raise OptionError(f"{name} must be a non-negative number")
    pts = read_gpx(track) if isinstance(track, (str, os.PathLike)) else np.asarray(track, float)
    if len(pts) < 2:
        raise VertmaxxerError("The route has no track points")
    dem = _dem_for([tuple(pts[0]), tuple(pts[-1])], dem)
    L0 = _along(pts)[-1]
    extra = budget_mi * MI_TO_M - L0 if budget_mi is not None else extra_mi * MI_TO_M
    reach = max(extra, 0) / 2 + 300  # side trips go out and back
    closed_loop = core._haversine(*pts[0], *pts[-1]) < 100
    centers = pts[np.linspace(0, len(pts) - 1, max(2, int(L0 / max(reach, 500)) + 2)).astype(int)]

    print(f"Route {core.show_d(L0)}; up to {core.show_d(extra)} more; fetching trails and peaks...")
    osm = core.fetch_osm([tuple(c) for c in centers], reach + 200, True)
    q = "[out:json][timeout:180];(" + "".join(
        f'node["natural"="peak"]["name"](around:{reach + 200:.0f},{la:.6f},{lo:.6f});' for la, lo in centers) + ");out;"
    peaks = {x["id"]: (x["lat"], x["lon"], x["tags"]["name"]) for x in core._overpass(q, "peaks")["elements"]}
    anchors = [tuple(pts[0]), tuple(pts[-1])] + [(la, lo) for la, lo, _ in peaks.values()]
    closures = core.load_closures(sorted(core.CLOSURES_DIR.glob("*.json")))
    # The route's ends snap to a street when one is nearer than any trail, as in find_route.
    street = [k for k in (0, 1) if _nearer_street(osm, anchors[k])]
    raw, ids = core.build_graph(osm, True, None, anchors, closed=closures, report=1 if closed_loop else 2,
                                snap_roads=street)
    raw = [e for e in raw if not e.get("major")]  # highways only where they meet other ways
    at = {}
    for e in raw:
        at[e["u"]], at[e["v"]] = e["latlon"][0], e["latlon"][-1]
    summit = {n: name for n, (la, lo, name) in zip(ids[2:], peaks.values())
              if n in at and core._haversine(*at[n], la, lo) <= summit_m}
    start, end = ids[0], (ids[0] if closed_loop else ids[1])
    edges = core.contract([dict(e) for e in raw], {start, end} | set(summit))
    core.add_elevation(edges, 50.0, dem)
    edges = _collapse_parallel(core.subdivide(edges, None))  # per-edge climb, no splitting

    base = _match(edges, pts, match_m, start, end)
    L_base = sum(e["length"] * c for e, c in zip(edges, base))
    if abs(L_base - L0) > 0.05 * L0:
        print(f"Warning: the route matched {core.show_d(L_base)} of trail for a {core.show_d(L0)} track; "
              "try a larger match distance.", file=sys.stderr)
    budget = budget_mi * MI_TO_M if budget_mi is not None else L_base + extra_mi * MI_TO_M
    if budget < L_base:
        raise VertmaxxerError(f"The budget is shorter than the route itself ({core.show_d(L_base)} of trail)")
    # The route's own turnarounds stay allowed; anything new turns around only at a summit.
    G = nx.MultiGraph()
    for e, c in zip(edges, base):
        if c:
            G.add_edge(e["u"], e["v"])
    tips = {n for n in G if G.degree(n) == 1}
    for e, c in zip(edges, base):
        if c:
            e["require"] = int(c)
        else:
            e["spur_only"] = True
    shape = dict(loops=None, spurs=None, start=None, reuse=True)
    m, s, t, proven = core.solve(edges, [start], None if closed_loop else [end], budget, shape, 0, time_limit_s,
                                 workers, False, hint=base, no_turnarounds=True, turnaround_ok=set(summit) | tips)
    route = _route(edges, np.asarray(m), s, t, proven, 0.0, 0.0, anchors=anchors)
    if closed_loop and _runs_backward(route, pts):
        route = _reversed_route(route)
    base_route = core.assemble(edges, base, start, end)

    # Side trips: the added edges (all run out and back), grouped by where they leave the route.
    H = nx.Graph()
    for k in range(len(edges)):
        if m[k] and not base[k]:
            H.add_edge(edges[k]["u"], edges[k]["v"], k=k)
    d_base = _along(pts)
    lat0 = np.radians(pts[:, 0].mean())
    trips = []
    for comp in nx.connected_components(H):
        ks = [H[u][v]["k"] for u, v in H.subgraph(comp).edges]
        junction = next((n for n in comp if n in G), None)
        mi = float("nan")
        if junction is not None:
            la, lo = at[junction]
            mi = d_base[np.argmin(np.hypot((pts[:, 0] - la) * 110540, (pts[:, 1] - lo) * 111320 * np.cos(lat0)))]
            mi /= MI_TO_M
        trips.append(SideTrip(leaves_at_mi=float(mi), length_mi=2 * sum(edges[k]["length"] for k in ks) / MI_TO_M,
                              gain_ft=sum(edges[k]["var"] for k in ks) * M_TO_FT,  # out and back: all |dz| climbed once
                              summits=[summit[n] for n in comp if n in summit]))
    trips.sort(key=lambda t: t.leaves_at_mi)
    return Spurred(route=route, base_gain_ft=float(np.sum(np.maximum(0, np.diff(base_route["z"]))) * M_TO_FT),
                   base_distance_mi=L_base / MI_TO_M, side_trips=trips)
