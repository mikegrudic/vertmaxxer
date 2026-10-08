"""Python API: find a route, or add summit side trips to one. The command-line tools are thin wrappers over these."""
import json
import re
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from . import core
from .core import MI_TO_M, M_TO_FT, VertmaxxerError


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

    def write_gpx(self, path, name=None):
        core.write_gpx(path, vars(self), name or f"{self.shape} {self.distance_mi:.1f} mi")

    def plot(self, path):
        """Map over the searched network, and elevation profile (needs matplotlib)."""
        core.plot(path, self.edges, vars(self), self.anchors)


def _route(edges, m, start, end, proven, min_loop, min_loop_frac, heads=None, anchors=None):
    r = core.assemble(edges, m, start, end)
    shape, details = core.classify(edges, m, start, end, min_loop, min_loop_frac)
    return Route(lat=r["lat"], lon=r["lon"], z=r["z"], z_raw=r["z_raw"], dist=r["dist"], legs=r["legs"], shape=shape,
                 details=details, proven=bool(proven), end=(heads or {}).get(end), edges=edges, anchors=anchors)


def _nearer_street(osm, point):
    """Whether a street is nearer ``point`` than any trail (a start in town, say, rather than at a trailhead)."""
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm["elements"] if el["type"] == "node"}
    best = {True: np.inf, False: np.inf}
    for w in osm["elements"]:
        tags = w.get("tags", {})
        if w["type"] != "way" or not core._usable(tags, True, None) or core._not_a_start(tags):
            continue
        ll = np.array([nodes[n] for n in w["nodes"] if n in nodes])
        if len(ll):
            road = core._is_road(tags["highway"])
            best[road] = min(best[road], float(core._haversine(*point, ll[:, 0], ll[:, 1]).min()))
    return best[True] < best[False]


def _points(p):
    """One (lat, lon) or a list of them, as a list."""
    return [tuple(map(float, p))] if np.ndim(p) == 1 else [tuple(map(float, q)) for q in p]


def find_route(start, distance_mi=None, topology="lollipop", *, time_h=None, pace=None, gap=None, end=None, end_trailheads=False, min_end_dist_mi=1.0,
               max_road_fraction=0.1, trailhead_roads_m=400.0, roads=False, roads_only=False, paved_only=False,
               road_time_frac=None, ways=None, closures=(), any_end=False, minimize=False, min_loop_mi=1.0,
               min_loop_frac=0.25, max_sac=None, dem="3dep", smooth_m=50.0, seg_max_m=500.0, time_limit_s=120.0,
               workers=8, verbose=False):
    """The route with the most climbing (or with ``minimize``, the least) from ``start``.

    ``start`` is (lat, lon), or a list of them to let the solver pick. ``distance_mi`` is the most the route may
    run; or give ``time_h`` with ``pace`` (minutes per mile, which just sets the distance) or ``gap`` (grade-adjusted
    minutes per mile: each stretch then costs its Strava GAP-equivalent flat distance; a stretch run once counts the
    average of its two directions). ``topology`` is one of ``core.TOPOLOGIES``; ``traverse`` needs ``end`` (a point or list) or
    ``end_trailheads``. Roads: by default up to ``max_road_fraction`` of the distance, besides road crossings and
    the roads within ``trailhead_roads_m`` of the start; ``roads`` allows any amount; ``roads_only`` uses streets
    alone. ``ways`` is a dict (or JSON path) of OSM way ids, {"include": [...], "exclude": [...]}. ``closures`` are
    JSON files of closed segments, besides the bundled ones. ``max_sac`` (1-6) skips harder trails.

    Raises VertmaxxerError if no route fits, or a data source fails.
    """
    budget_flat = None  # grade-adjusted (flat-equivalent) m, with a time at a GAP
    if time_h is not None:
        if distance_mi is not None:
            raise ValueError("give a distance or a time, not both")
        if (pace is None) == (gap is None):
            raise ValueError("with a time, give one of pace or gap (minutes per mile)")
        if pace is not None:
            distance_mi = time_h * 60 / pace
        elif minimize:
            raise ValueError("minimize needs a distance, or a time with a pace")
        else:
            budget_flat = time_h * 60 / gap * MI_TO_M
            g = np.linspace(0, 50, 501)
            distance_mi = budget_flat / MI_TO_M / np.min((core.gap_factor(g) + core.gap_factor(-g)) / 2)
    elif distance_mi is None:
        raise ValueError("give distance_mi, or time_h with pace or gap")
    if topology not in core.TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; one of {', '.join(core.TOPOLOGIES)}")
    shape = core.TOPOLOGIES[topology]
    starts_ll = _points(start)
    ends_ll = _points(end) if end is not None else []
    p2p = bool(ends_ll) or end_trailheads
    if ends_ll and end_trailheads:
        raise ValueError("give either end or end_trailheads")
    if p2p and topology not in ("traverse", "any"):
        raise ValueError(f"{topology} routes return to the start; use traverse or any for point-to-point")
    if topology == "traverse" and not p2p:
        raise ValueError("traverse needs end or end_trailheads")
    if roads_only and end_trailheads:
        raise ValueError("end_trailheads finds trail ends; with roads_only give end")
    if isinstance(ways, str):
        ways = json.load(open(ways))
    ways = ways or {}
    include, exclude = set(ways.get("include", ())), set(ways.get("exclude", ()))

    budget = distance_mi * MI_TO_M
    anchors = starts_ll + ends_ll
    max_sac = None if max_sac is None else max_sac - 1
    any_roads = roads or roads_only or road_time_frac is not None
    closed = core.load_closures(sorted(core.CLOSURES_DIR.glob("*.json")) + list(closures))
    heads = {}
    if end_trailheads:  # the route may end anywhere within the full budget of the start
        osm = core.fetch_osm(starts_ll, budget, True)
        heads = core.road_trailheads(osm, starts_ll, budget, roads, max_sac, closed)
        heads = {n: h for n, h in heads.items()
                 if min(core._haversine(h[0], h[1], la, lo) for la, lo in starts_ll) >= min_end_dist_mi * MI_TO_M
                 and (any_end or (h[2].split(" at ")[-1] not in core.EXCLUDED_END_ROADS
                                  and all(core._haversine(h[0], h[1], la, lo) > r for la, lo, r in core.EXCLUDED_ENDS)))}
        print(f"{len(heads)} trailheads at least {min_end_dist_mi:g} mi from the start")
    else:
        osm = core.fetch_osm(anchors, budget / 2, True)
    if exclude:
        osm = dict(osm, elements=[el for el in osm["elements"] if not (el["type"] == "way" and el["id"] in exclude)])
    # A start nearer a street than a trail starts on the street; the walk from it to the trails is a road walk.
    street = (list(range(len(anchors))) if roads_only
              else [k for k, p in enumerate(starts_ll) if _nearer_street(osm, p)])
    edges, anchor_ids = core.build_graph(osm, True, max_sac, anchors, extra_ids=heads, closed=closed, snap_roads=street)
    if any_roads:  # major roads only where they meet others
        edges = [e for e in edges if not e.get("major")]
    else:  # trails, and road walks between them; crossings and the start's own roads are free
        free = {id(e) for e in core.road_connectors(edges, core.CROSSING_M) if e["road"]}
        free |= core.trailhead_roads(edges, anchor_ids[: len(starts_ll)], trailhead_roads_m)
        minor = [e for e in edges if not e.get("major")]
        walks = {id(e) for e in core.road_connectors(minor, max_road_fraction * budget)}
        walks |= core.trailhead_roads(edges, [anchor_ids[k] for k in street], max_road_fraction * budget)
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
    edges = core.contract(core.prune(edges, budget, starts, ends), set(anchor_ids) | set(heads))
    core.add_elevation(edges, smooth_m, dem)
    edges = core.prune(core.subdivide(edges, None if shape["spurs"] == 0 else seg_max_m), budget, starts, ends)
    if budget_flat is not None:
        for e in edges:
            e["cost"] = sum(e["gap"]) / 2

    m, s, t, proven = core.solve(edges, starts, ends, budget_flat or budget, shape, min_loop_mi * MI_TO_M, time_limit_s, workers,
                                 verbose, min_loop_frac=min_loop_frac, road_time_frac=road_time_frac,
                                 minimize=minimize, max_road_frac=None if any_roads else max_road_fraction,
                                 min_length=0.98 * budget if minimize else 0.0)
    return _route(edges, m, s, t, proven, min_loop_mi * MI_TO_M, min_loop_frac, heads, anchors)


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
    """Track (or route) points of a GPX file, as an (n, 2) array of lat, lon."""
    t = open(path).read()
    p = re.findall(r'<(?:trkpt|rtept)[^>]*lat="([-\d.]+)"[^>]*lon="([-\d.]+)"', t)
    p = p or [(a, b) for b, a in re.findall(r'<(?:trkpt|rtept)[^>]*lon="([-\d.]+)"[^>]*lat="([-\d.]+)"', t)]
    return np.array(p, float)


def _along(pts):
    lat0 = np.radians(pts[:, 0].mean())
    return np.r_[0, np.cumsum(np.hypot(np.diff(pts[:, 1]) * 111320 * np.cos(lat0), np.diff(pts[:, 0]) * 110540))]


def _match(edges, pts, tol, start, end):
    """Times the track runs each edge: the junctions it passes within ``tol`` (m), in order, from ``start`` to
    ``end``, joined by the shortest trail between them, so the result is always one continuous walk."""
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
    seq = [start]
    for p in trk:
        dd = np.hypot(*(nxy - p).T)
        i = int(dd.argmin())
        if dd[i] <= tol and seq[-1] != nodes[i]:
            seq.append(nodes[i])
    if seq[-1] != end:
        seq.append(end)
    counts = np.zeros(len(edges), int)
    for a, b in zip(seq, seq[1:]):
        path = nx.shortest_path(G, a, b, weight="w")
        for u, v in zip(path, path[1:]):
            counts[min(G[u][v], key=lambda k: G[u][v][k]["w"])] += 1
    if (counts > 2).any():
        print(f"Warning: the route runs {(counts > 2).sum()} trail sections more than twice; counted as twice.")
    return np.minimum(counts, 2)


def spurify(track, extra_mi=None, budget_mi=None, *, time_limit_s=60.0, workers=8, match_m=15.0, summit_m=60.0):
    """Add out-and-back side trips to named summits to a route, keeping the route itself.

    ``track`` is a GPX path or an (n, 2) array of lat, lon. Give ``extra_mi`` (miles that may be added) or
    ``budget_mi`` (total miles). Side trips turn around only at peaks within ``summit_m`` of a trail (or where the
    route already turns around); ``match_m`` is how closely the track must follow the trails.
    """
    if (extra_mi is None) == (budget_mi is None):
        raise ValueError("give one of extra_mi or budget_mi")
    pts = read_gpx(track) if isinstance(track, str) else np.asarray(track, float)
    L0 = _along(pts)[-1]
    budget = budget_mi * MI_TO_M if budget_mi is not None else L0 + extra_mi * MI_TO_M
    reach = max(budget - L0, 0) / 2 + 300  # side trips go out and back
    closed_loop = core._haversine(*pts[0], *pts[-1]) < 100
    centers = pts[np.linspace(0, len(pts) - 1, max(2, int(L0 / max(reach, 500)) + 2)).astype(int)]

    print(f"Route {L0 / MI_TO_M:.2f} mi; budget {budget / MI_TO_M:.2f} mi; fetching trails and peaks...")
    osm = core.fetch_osm([tuple(c) for c in centers], reach + 200, True)
    q = "[out:json][timeout:180];(" + "".join(
        f'node["natural"="peak"]["name"](around:{reach + 200:.0f},{la:.6f},{lo:.6f});' for la, lo in centers) + ");out;"
    peaks = {x["id"]: (x["lat"], x["lon"], x["tags"]["name"]) for x in core._overpass(q, "peaks")["elements"]}
    anchors = [tuple(pts[0]), tuple(pts[-1])] + [(la, lo) for la, lo, _ in peaks.values()]
    closures = core.load_closures(sorted(core.CLOSURES_DIR.glob("*.json")))
    raw, ids = core.build_graph(osm, True, None, anchors, closed=closures)
    raw = [e for e in raw if not e.get("major")]  # highways only where they meet other ways
    at = {}
    for e in raw:
        at[e["u"]], at[e["v"]] = e["latlon"][0], e["latlon"][-1]
    summit = {n: name for n, (la, lo, name) in zip(ids[2:], peaks.values())
              if n in at and core._haversine(*at[n], la, lo) <= summit_m}
    start, end = ids[0], (ids[0] if closed_loop else ids[1])
    edges = core.contract([dict(e) for e in raw], {start, end} | set(summit))
    core.add_elevation(edges, 50.0, "3dep")
    edges = core.subdivide(edges, None)  # per-edge climb, no splitting

    base = _match(edges, pts, match_m, start, end)
    L_base = sum(e["length"] * c for e, c in zip(edges, base))
    if abs(L_base - L0) > 0.05 * L0:
        print(f"Warning: the route matched {L_base / MI_TO_M:.2f} mi of trail for a {L0 / MI_TO_M:.2f} mi track; "
              "try a larger match distance.")
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
    base_route = core.assemble(edges, base, start, end)

    # Side trips: the added edges (all run out and back), grouped by where they leave the route.
    H = nx.Graph()
    for k in range(len(edges)):
        if m[k] and not base[k]:
            H.add_edge(edges[k]["u"], edges[k]["v"], k=k)
    d_base = _along(pts)
    lat0 = np.radians(pts[:, 0].mean())
    trips = []
    for comp in sorted(nx.connected_components(H), key=lambda c: -len(c)):
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
    return Spurred(route=route, base_gain_ft=float(np.sum(np.maximum(0, np.diff(base_route["z"]))) * M_TO_FT),
                   base_distance_mi=L_base / MI_TO_M, side_trips=trips)
