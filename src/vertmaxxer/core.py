"""Find the trail route with maximum vertical gain from given trailhead(s),
subject to a distance budget and a route topology.

Trails (and optionally roads) come from OpenStreetMap via Overpass; elevation
comes from USGS 3DEP (US only), directly or via AWS Terrain Tiles. Both are cached on disk.

Model
-----
Each edge of the junction graph is traversed 0, 1 or 2 times. A closed walk
loses exactly what it gains, so its total gain is half the summed |dz| of the
traversed edges regardless of direction. The objective is therefore linear in
the traversal counts, and any connected choice of counts with even degree at
every node is realizable as a walk (Euler). A point-to-point route adds
(z_end - z_start) / 2. The model is solved with OR-Tools CP-SAT; connectivity
is a rooted spanning arborescence with integer ranks.

The solver finds near-optimal routes quickly but its upper bound is weak (the
LP relaxation can take fractions of long routes), so "bound" overstates the
remaining headroom and runs usually stop at the time limit rather than with a
proof of optimality.

Topologies are exact shapes; a route that fits a simpler one (e.g. a lollipop
whose loop shrinks to nothing) is not reported under the more complex name.
Loops are run once and are at least --min-loop long and --min-loop-frac of the
route; only stems and bridges are retraced.
  loop             a single loop through the trailhead
  out-and-back     out and back along the same trail
  lollipop         a retraced stem from the trailhead to one loop
  double-lollipop  a retraced stem from the trailhead to two separate loops
  dumbbell         start on loop A, cross a retraced bridge, run loop B, cross back, finish A
  figure-8         two loops sharing one crossing point, nothing retraced
  traverse         point-to-point simple path, to --end or (--end-trailheads) any
                   other trailhead, i.e. any point where a trail meets a paved road open to cars
  any              anything with <= 2 traversals per edge
The shape of the solved route is classified independently and printed.

When spurs are allowed, long edges are split every --seg-max meters so
turnarounds can fall mid-trail.
"""

import hashlib
import io
import json
import heapq
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import networkx as nx
import numpy as np
import requests
from ortools.sat.python import cp_model
from PIL import Image
from scipy.ndimage import map_coordinates



class VertmaxxerError(Exception):
    """No route, or a data source failed."""


class OptionError(ValueError):
    """Inconsistent or invalid options."""


CACHE_DIR = Path(os.environ.get("VERTMAXXER_CACHE", Path.home() / ".cache" / "vertmaxxer"))
OVERPASS_WAITS = [5, 10, 20, 30, 45, 60, 60, 60]  # s after each failed attempt: about five minutes in all
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.openstreetmap.fr/api/interpreter"]
_DEAD_MIRRORS = set()  # mirrors that refused us or answered with an internal error this run
DEM_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
DEM_RES_DEG = 1.0 / 3 / 3600  # 1/3 arc-second, ~10 m
DEM_TILE_PX = 1000  # larger requests often time out at the USGS server
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRARIUM_ZOOM = 14
EARTH_RADIUS_M = 6371000.0
MI_TO_M = 1609.344
M_TO_FT = 3.280839895
RESAMPLE_M = 10.0
SNAP_M = 500.0
SNAP_WARN_M, SNAP_FAIL_M = 200.0, 1000.0
STREET_SNAP_M = 100.0  # a street start snaps to the biggest street network this close (not a cut-off lane)
FETCH_PAD_M = 1000.0  # downloads reach this much further than asked, for reuse  # a start or end this far from any usable way is suspect, or an error
GAP_M = 10.0  # join a trail's dead end to another trail this close: an OSM digitizing gap
GAP_SAME_NAME_M = 50.0  # ... or to a trail of the same name this close
GAP_TABLE = Path(__file__).parent / "data" / "strava_GAP_table.dat"
MIN_TRAIL_NET = 1609.344  # m of connected trail for a network's road crossings to count as access points
ROAD_GAP_MIN = 0.75  # lowest Strava GAP factor on a plausible road grade; bounds road distance from road time
HEADERS = {"User-Agent": "vertmaxxer/0.1 (github.com/mikegrudic/vertmaxxer)"}  # the fr mirror refuses some words

TRAIL_HIGHWAYS = ["path", "footway", "track", "bridleway", "steps", "cycleway"]
ROAD_HIGHWAYS = ["residential", "unclassified", "tertiary", "secondary", "service", "living_street", "road"]
# Roads whose junctions with trails count as trailheads (wider than the set a route may follow with --roads).
MAJOR_ROADS = ["primary", "primary_link", "trunk", "trunk_link"]  # fetched with roads, but only ever crossed
JUNCTION_ROADS = ROAD_HIGHWAYS + ["primary", "primary_link", "secondary_link", "tertiary_link", "trunk", "trunk_link"]
UNPAVED = {"unpaved", "gravel", "fine_gravel", "compacted", "dirt", "earth", "ground", "grass", "mud", "sand",
           "pebblestone", "rock", "woodchips"}
SAC_SCALE = [
    "hiking", "mountain_hiking", "demanding_mountain_hiking",
    "alpine_hiking", "demanding_alpine_hiking", "difficult_alpine_hiking",
]

# loops: exact number of loops (None: any). spurs: max dead-end turnarounds (None: any).
# start: whether the trailhead lies on a loop or at the end of a retraced stem (None: either).
# reuse: edges may be traversed twice. retrace_loops: whole loops may be traversed twice.
TOPOLOGIES = {
    "loop": dict(loops=1, spurs=0, start="loop", reuse=False),
    "out-and-back": dict(loops=0, spurs=1, start="stem", reuse=True),
    "lollipop": dict(loops=1, spurs=0, start="stem", reuse=True),
    "double-lollipop": dict(loops=2, spurs=0, start="stem", reuse=True),
    "dumbbell": dict(loops=2, spurs=0, start="loop", reuse=True),
    "figure-8": dict(loops=2, spurs=0, start="loop", reuse=False),
    "traverse": dict(loops=0, spurs=0, start=None, reuse=False),
    "loop-spurs": dict(loops=1, spurs=None, start=None, reuse=True),  # one loop plus any out-and-back side trips
    "spurred": dict(loops=None, spurs=None, start=None, reuse=True),  # any closed route with side trips (to summits)
    "any": dict(loops=None, spurs=None, start=None, reuse=True, retrace_loops=True),
}


def gap_factor(grade_percent):
    """Strava grade-adjusted-pace factor (time per meter relative to flat), from the gap calculator's table."""
    if not hasattr(gap_factor, "table"):
        gap_factor.table = np.loadtxt(GAP_TABLE).T
    g, f = gap_factor.table
    return np.interp(grade_percent, g, f)


def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def _cached(name, fetch, load, save, keep=lambda data: True):
    """``load`` the cached file, or ``fetch`` and ``save`` it (if ``keep(data)``). Files are written whole or not
    at all, and an unreadable one (e.g. from an older, interrupted run) is fetched again."""
    path = CACHE_DIR / name
    if path.exists():
        try:
            return load(path)
        except Exception:
            print(f"Cached {name} is unreadable; fetching it again")
            path.unlink()
    data = fetch()
    if not keep(data):
        return data
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")  # same suffix: np.save keeps the name
    try:
        save(tmp, data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return data


# ---------------------------------------------------------------- OSM graph

def fetch_osm(points, radius_m, roads, way_ids=()):
    """Trails (and roads) within ``radius_m`` of the points, plus the ways in ``way_ids`` whatever their type.

    From the cache when a download covers it: the exact query, or else the smallest download whose circles contain
    these (same or more roads, the same extra ways or more), clipped to them. New downloads reach FETCH_PAD_M
    further, so a later start nearby or a slightly longer route reuses them."""
    points = [tuple(map(float, p)) for p in points]

    def query(r):
        highways = "|".join(dict.fromkeys(TRAIL_HIGHWAYS + (ROAD_HIGHWAYS + MAJOR_ROADS if roads else [])))
        clauses = "".join(f'way["highway"~"^({highways})$"](around:{r:.0f},{la:.6f},{lo:.6f});' for la, lo in points)
        if way_ids:
            clauses += f"way(id:{','.join(map(str, sorted(way_ids)))});"
        return f"[out:json][timeout:300];({clauses});(._;>;);out body;"

    exact = CACHE_DIR / f"osm_{_key(query(radius_m))}.json"
    if exact.exists():
        return _overpass(query(radius_m), "osm")
    best = None
    for meta in CACHE_DIR.glob("osm_*.meta.json"):
        try:
            m = json.loads(meta.read_text())
            if (m["roads"] >= bool(roads) and set(m["way_ids"]) >= set(way_ids)
                    and all(any(_haversine(*p, *q) + radius_m <= m["radius"] for q in m["points"]) for p in points)
                    and (best is None or m["radius"] < best[0])):
                best = (m["radius"], meta.with_name(meta.name.replace(".meta.json", ".json")))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if best is not None:
        try:
            return _clip(json.loads(best[1].read_text()), points, radius_m, way_ids)
        except (OSError, ValueError):
            pass
    padded = radius_m + FETCH_PAD_M
    data = _overpass(query(padded), "osm")
    here = CACHE_DIR / f"osm_{_key(query(padded))}.json"
    if here.exists():
        meta = dict(points=[list(p) for p in points], radius=float(padded), roads=bool(roads), way_ids=sorted(way_ids))
        here.with_name(here.name.replace(".json", ".meta.json")).write_text(json.dumps(meta))
    return _clip(data, points, radius_m, way_ids)


def _clip(data, points, radius_m, way_ids=()):
    """The ways of an Overpass answer with a node within ``radius_m`` of a point (or listed in ``way_ids``), and their
    nodes."""
    nodes = {e["id"]: e for e in data["elements"] if e["type"] == "node"}
    ids = np.array(list(nodes))
    ll = np.array([(nodes[n]["lat"], nodes[n]["lon"]) for n in ids]).reshape(-1, 2)
    near = np.zeros(len(ids), bool)
    for la, lo in points:
        near |= _haversine(la, lo, ll[:, 0], ll[:, 1]) <= radius_m
    inside = set(ids[near].tolist())
    ways = [e for e in data["elements"] if e["type"] == "way"
            and (e["id"] in way_ids or any(n in inside for n in e["nodes"]))]
    used = {n for w in ways for n in w["nodes"]}
    return dict(data, elements=[nodes[n] for n in used if n in nodes] + ways)


def _key(query):
    return hashlib.sha1(query.encode()).hexdigest()[:16]


def _overpass(query, prefix):
    """Run an Overpass query, trying each mirror, with the result cached on disk."""
    def fetch():
        print("Querying Overpass...")
        timeouts = 0
        for attempt, wait in enumerate(OVERPASS_WAITS):
            live = [u for u in OVERPASS_URLS if u not in _DEAD_MIRRORS] or OVERPASS_URLS
            url = live[attempt % len(live)]
            try:
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=360)
                if r.status_code in (403, 500) and url != OVERPASS_URLS[0]:
                    _DEAD_MIRRORS.add(url)
                if r.status_code == 400:  # the query itself is bad; retrying won't help
                    raise VertmaxxerError(f"Overpass rejected the query: {r.text.strip()[:300]}")
                if r.status_code == 429:  # rate limited: give it longer
                    wait = max(wait, 60)
                r.raise_for_status()
                data = r.json()
                # Server-side timeouts and out-of-memory errors come back as 200 with partial data and a remark.
                remark = data.get("remark", "")
                if "out of memory" in remark:
                    raise VertmaxxerError("The query is too big for the Overpass server; try a shorter distance")
                if "runtime error" in remark or "timed out" in remark:
                    timeouts += 1
                    if timeouts > 2:
                        raise VertmaxxerError("Overpass keeps timing out on this query; try a shorter distance")
                    raise requests.RequestException(remark)
                return data
            except (requests.RequestException, ValueError) as err:
                print(f"  {url}: {err}; retrying in {wait} s")
                time.sleep(wait)
        raise VertmaxxerError("All Overpass servers failed; try again later.")

    return _cached(f"{prefix}_{_key(query)}.json", fetch, lambda p: json.loads(p.read_text()),
                   lambda p, d: p.write_text(json.dumps(d)),
                   keep=lambda d: bool(d.get("elements")) or prefix != "osm")  # no peaks nearby is an answer


def _drivable(tags):
    """A road counts for trailheads unless tagged unpaved or closed to cars (untagged roads count)."""
    cars = tags.get("motor_vehicle", tags.get("access"))
    return tags.get("surface") not in UNPAVED and cars not in ("no", "private")


def road_trailheads(osm, points, radius_m, roads, max_sac, closed=frozenset()):
    """Trailheads, taken as every OSM node shared by a usable trail and a paved road open to cars (through a
    trail segment that isn't in ``closed``).

    Returns {node id: (lat, lon, label)}.
    """
    clauses = "".join(
        f'way["highway"~"^({"|".join(JUNCTION_ROADS)})$"](around:{radius_m:.0f},{lat:.6f},{lon:.6f});'
        for lat, lon in points
    )
    road_ways = _overpass(f"[out:json][timeout:300];({clauses});out body;", "roads")["elements"]
    road_name = {}
    for w in road_ways:
        tags = w.get("tags", {})
        if not _drivable(tags):
            continue
        for n in w["nodes"]:
            road_name.setdefault(n, tags.get("name") or tags.get("ref") or f"a {tags['highway']} road")
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm["elements"] if el["type"] == "node"}
    heads = {}
    trails = [w for w in osm["elements"] if w["type"] == "way" and w.get("tags", {}).get("highway") in TRAIL_HIGHWAYS
              and _usable(w["tags"], roads, max_sac)]
    for w in _drop_closed(trails, closed) if closed else trails:
        tags = w["tags"]
        for n in w["nodes"]:
            if n in road_name and n not in heads:
                heads[n] = (*nodes[n], f"{tags.get('name') or tags['highway']} at {road_name[n]}")
    return heads


def _is_road(highway):
    return highway in ROAD_HIGHWAYS or highway in MAJOR_ROADS


def _not_a_start(tags):
    """Roads a street start doesn't snap to: highways, and driveways and parking aisles."""
    return tags.get("highway") in MAJOR_ROADS or tags.get("service") in NOT_ROUTES


def _unmarked(tags):
    """Herd paths and other informal or hard-to-follow ways."""
    return tags.get("informal") == "yes" or tags.get("trail_visibility") in ("bad", "horrible", "no")


def _usable(tags, roads, max_sac, marked_only=False):
    if marked_only and _unmarked(tags):
        return False
    foot = tags.get("foot")
    if foot in ("no", "private"):
        return False
    if tags.get("access") in ("no", "private") and foot not in ("yes", "designated", "permissive"):
        return False
    if tags.get("area") == "yes":
        return False
    if tags.get("footway") in ("sidewalk", "crossing"):  # the road itself covers these
        return False
    sac = tags.get("sac_scale")
    if max_sac is not None and sac in SAC_SCALE and SAC_SCALE.index(sac) > max_sac:
        return False
    return True


def build_graph(osm, roads, max_sac, anchors, extra_ids=(), connectors_m=0.0, closed=frozenset(), snap_roads=(),
                include=frozenset(), report=None, marked_only=False):
    """Split usable ways at junctions (and at ``extra_ids``) and snap anchor points to the nearest way node.

    With ``connectors_m`` > 0 (and ``osm`` fetched with roads), road stretches of at most that length
    joining two trails are kept as connectors, e.g. to cross a road between adjacent trailheads.
    ``closed`` holds OSM node pairs (frozensets) of closed segments, which are dropped. Anchors whose
    index is in ``snap_roads`` snap to the nearest road node instead (for starts on a street, e.g. a house).
    Ways in ``include`` are used whatever their tags. Only the first ``report`` anchors' snaps are printed.

    Returns (edges, anchor_ids), where each edge is a dict with OSM node ids
    ``u``, ``v``, polyline ``latlon`` (k, 2), per-vertex trail ``names``,
    ``length`` (m) and ``road``.
    """
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm["elements"] if el["type"] == "node"}
    # Included ways count as footpaths when they have no highway tag (a pier, a plaza).
    ways = [dict(el, tags={"highway": "footway", **el.get("tags", {})}) if el["id"] in include else el
            for el in osm["elements"] if el["type"] == "way"]
    ways = [w for w in ways if "highway" in w.get("tags", {})
            and (w["id"] in include or _usable(w["tags"], roads, max_sac, marked_only))]
    if not ways:
        raise VertmaxxerError("No usable ways found near the trailhead(s).")
    if closed:
        ways = _drop_closed(ways, closed)

    anchor_ids = []
    for k, (lat, lon) in enumerate(anchors):
        on_road = k in snap_roads
        snap_ways = [w for w in ways if _is_road(w["tags"]["highway"]) == on_road
                     and not (on_road and _not_a_start(w["tags"]))]
        way_nodes = np.array(sorted({n for w in snap_ways for n in w["nodes"]}))
        if not len(way_nodes):
            raise VertmaxxerError(f"No {'streets' if on_road else 'trails'} near {lat:.5f}, {lon:.5f}")
        way_ll = np.array([nodes[n] for n in way_nodes])
        # Snap to the nearest node of the biggest trail network within SNAP_M, so a stub path at a
        # parking lot or campground doesn't capture the start.
        G = nx.Graph()
        for w in snap_ways:
            nx.add_path(G, w["nodes"])
        comp_size = {}
        for comp in nx.connected_components(G):
            for n in comp:
                comp_size[n] = len(comp)
        d = _haversine(lat, lon, way_ll[:, 0], way_ll[:, 1])
        near = np.flatnonzero(d <= (STREET_SNAP_M if on_road else SNAP_M))
        if len(near):
            biggest = max(comp_size[way_nodes[i]] for i in near)
            i = int(min((i for i in near if comp_size[way_nodes[i]] == biggest), key=lambda i: d[i]))
        else:
            i = int(np.argmin(d))
        tags = next(iter(f" ({w['tags'].get('name', w['tags']['highway'])})" for w in snap_ways if way_nodes[i] in w["nodes"]))
        if report is None or k < report:
            size = "" if on_road else f", in a trail network of {comp_size[way_nodes[i]]} nodes"
            print(f"({lat}, {lon}) snapped {d[i]:.0f} m to OSM node {way_nodes[i]}{tags}{size}")
        if report is None or k < report:
            if d[i] > SNAP_FAIL_M:
                raise VertmaxxerError(f"Nothing walkable within {SNAP_FAIL_M / 1000:g} km of {lat:.5f}, {lon:.5f}")
            if d[i] > SNAP_WARN_M:
                print(f"Warning: {lat:.5f}, {lon:.5f} is {d[i]:.0f} m from the nearest usable way; the route starts "
                      "or ends there instead", file=sys.stderr)  # stderr: quiet runs still show warnings
        anchor_ids.append(int(way_nodes[i]))

    count = Counter(n for w in ways for n in w["nodes"])
    gaps = _mapping_gaps(ways, nodes, count)
    junctions = {n for n, c in count.items() if c > 1} | set(anchor_ids) | set(extra_ids)
    junctions |= {w["nodes"][0] for w in ways} | {w["nodes"][-1] for w in ways}
    junctions |= {n for gap in gaps for n in gap}

    edges = []
    for w in ways:
        tags = w.get("tags", {})
        name = (tags.get("name") or tags.get("ref") or tags["highway"]) + (" (unmarked)" if _unmarked(tags) else "")
        seq = w["nodes"]
        start = 0
        for i in range(1, len(seq)):
            if seq[i] in junctions:
                ll = np.array([nodes[n] for n in seq[start : i + 1]])
                is_road = _is_road(tags["highway"])
                # Bridges and tunnels: the bare-earth DEM there is the ground below or above, not the way.
                off_ground = tags.get("bridge", "no") != "no" or tags.get("tunnel", "no") not in ("no", "culvert")
                edges.append(dict(
                    u=seq[start], v=seq[i], latlon=ll, names=np.full(len(ll), name, dtype=object),
                    roadpt=np.full(len(ll), is_road), flatpt=np.full(len(ll), off_ground),
                    length=float(np.sum(_haversine(ll[:-1, 0], ll[:-1, 1], ll[1:, 0], ll[1:, 1]))),
                    road=is_road, highway=tags["highway"], way=w["id"], surface=tags.get("surface"),
                    service=tags.get("service"), major=tags["highway"] in MAJOR_ROADS,
                ))
                start = i
    for a, b in gaps:
        ll = np.array([nodes[a], nodes[b]])
        edges.append(dict(u=a, v=b, latlon=ll, names=np.full(2, "gap in map data", dtype=object), roadpt=np.zeros(2, bool),
                          flatpt=np.zeros(2, bool),
                          length=max(float(_haversine(*ll[0], *ll[1])), 0.2), road=False))
    edges = [e for e in edges if e["length"] > 0.1]
    if not roads:
        edges = road_connectors(edges, connectors_m)
    return edges, anchor_ids


def _drop_closed(ways, closed):
    """Split ways around closed segments, dropping those segments."""
    out = []
    for w in ways:
        cur = [w["nodes"][0]]
        for a, b in zip(w["nodes"][:-1], w["nodes"][1:]):
            if frozenset((a, b)) in closed:
                if len(cur) > 1:
                    out.append(dict(w, nodes=cur))
                cur = [b]
            else:
                cur.append(b)
        if len(cur) > 1:
            out.append(dict(w, nodes=cur))
    return out


def load_closures(paths):
    """Closed OSM segments from closure files (JSON with "closed_segments": [[node, node], ...])."""
    return frozenset(frozenset(p) for path in paths for p in json.load(open(path))["closed_segments"])


CROSSING_M = 50.0  # road stretches this short joining two trails are crossings, not road walks
# Traverses may not end at the Mount Washington summit facilities (lat, lon, radius m) or on these roads.
EXCLUDED_ENDS = [(44.27060, -71.30330, 500.0)]
EXCLUDED_END_ROADS = {"Mount Washington Auto Road", "Breakneck Road"}
CLOSURES_DIR = Path(__file__).parent / "data" / "closures"

NOT_ROUTES = ("driveway", "parking_aisle", "drive-through")  # service roads nobody runs


def road_network(edges, include=frozenset(), link_m=30.0):
    """Roads only, paved or not (no trails, tracks, driveways or parking aisles), plus the ways in ``include`` whatever
    their type, each of their dead ends joined to the nearest road node within ``link_m``: the sidewalks the graph
    leaves out."""
    out = [e for e in edges if e.get("way") in include
           or (e["road"] and not e.get("major") and e.get("service") not in NOT_ROUTES)]
    pos, deg = {}, Counter()
    for e in out:
        pos[e["u"]], pos[e["v"]] = e["latlon"][0], e["latlon"][-1]
        deg[e["u"]] += 1
        deg[e["v"]] += 1
    road = sorted({n for e in out if e["road"] for n in (e["u"], e["v"])})
    xy = np.array([pos[n] for n in road])
    for e in list(out):
        if e["road"] or e.get("way") not in include:
            continue
        for n in (e["u"], e["v"]):
            if deg[n] == 1:
                d = _haversine(*pos[n], xy[:, 0], xy[:, 1])
                i = int(np.argmin(d))
                if d[i] <= link_m:
                    out.append(dict(u=n, v=road[i], latlon=np.array([pos[n], pos[road[i]]]),
                                    names=np.full(2, "sidewalk", dtype=object), roadpt=np.zeros(2, bool),
                                    flatpt=np.zeros(2, bool), length=max(float(d[i]), 0.2), road=False))
    return out


def trailhead_roads(edges, starts, max_m):
    """Ids of the roads (not highways or driveways) lying entirely within ``max_m`` of a start: its parking lots
    and access roads, or the streets from a start in town."""
    at = {}
    for e in edges:
        at[e["u"]], at[e["v"]] = e["latlon"][0], e["latlon"][-1]
    pts = [at[n] for n in starts if n in at]
    return {id(e) for e in edges if e["road"] and not e.get("major") and e.get("service") != "driveway"
            and any(np.all(_haversine(e["latlon"][:, 0], e["latlon"][:, 1], *p) <= max_m) for p in pts)}


def road_connectors(edges, max_m):
    """Keep only road stretches that connect two different trail access points by at most ``max_m``.

    One multi-source Dijkstra over the road graph labels each road node with its nearest trail node; a road
    edge joining two labels within ``max_m`` (via both labels) keeps itself and the paths back to them.
    """
    # Access points are nodes of trail networks with at least MIN_TRAIL_NET of trail, not village path stubs.
    T = nx.Graph()
    for e in edges:
        if not e["road"]:
            T.add_edge(e["u"], e["v"], w=e["length"])
    trail_nodes = set()
    for comp in nx.connected_components(T):
        if T.subgraph(comp).size(weight="w") >= MIN_TRAIL_NET:
            trail_nodes |= comp
    adj = defaultdict(list)
    for k, e in enumerate(edges):
        if e["road"]:
            adj[e["u"]].append((e["v"], e["length"], k))
            adj[e["v"]].append((e["u"], e["length"], k))
    dist, src, pred = {}, {}, {}
    heap = [(0.0, n, n) for n in adj if n in trail_nodes]
    heapq.heapify(heap)
    while heap:
        d, n, s0 = heapq.heappop(heap)
        if n in dist:
            continue
        dist[n], src[n] = d, s0
        for m, w, k in adj[n]:
            if m not in dist and d + w <= max_m:
                pred.setdefault(m, None)
                if pred[m] is None or d + w < pred[m][0]:
                    pred[m] = (d + w, n, k)
                heapq.heappush(heap, (d + w, m, s0))
    keep = set()

    def back(n):
        while n in pred and pred[n] is not None and pred[n][2] not in keep and dist.get(n, 0) > 0:
            keep.add(pred[n][2])
            n = pred[n][1]

    for k, e in enumerate(edges):
        u, v = e["u"], e["v"]
        if e["road"] and u in dist and v in dist and src[u] != src[v] and dist[u] + e["length"] + dist[v] <= max_m:
            keep.add(k)
            back(u)
            back(v)
    return [e for k, e in enumerate(edges) if not e["road"] or k in keep]


def _mapping_gaps(ways, nodes, count):
    """Pairs (trail dead end, node) on different ways closer than GAP_M, or than GAP_SAME_NAME_M when both are
    trails of the same name: digitizing gaps where a trail should continue but its ways don't share a node."""
    reach = max(GAP_M, GAP_SAME_NAME_M)
    if reach <= 0:
        return []
    cell = reach / 111000.0
    grid = defaultdict(list)
    owner, names = {}, defaultdict(set)
    for wi, w in enumerate(ways):
        for n in w["nodes"]:
            owner.setdefault(n, set()).add(wi)
            if w["tags"].get("name") and not _is_road(w["tags"]["highway"]):
                names[n].add(w["tags"]["name"])
            la, lo = nodes[n]
            grid[(int(la / cell), int(lo / cell))].append(n)
    ends = {n for w in ways if not _is_road(w["tags"]["highway"])
            for n in (w["nodes"][0], w["nodes"][-1]) if count[n] == 1}
    pairs = set()
    for a in ends:
        la, lo = nodes[a]
        gi, gj = int(la / cell), int(lo / cell)
        best = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for b in grid[(gi + di, gj + dj)]:
                    if owner[b] & owner[a]:
                        continue
                    d = float(_haversine(la, lo, *nodes[b]))
                    if (d <= GAP_M or (d <= GAP_SAME_NAME_M and names[a] & names[b])) and (best is None or d < best[0]):
                        best = (d, b)
        if best:
            pairs.add(tuple(sorted((a, best[1]))))
    return sorted(pairs)


def prune(edges, budget, starts, ends, partial=False):
    """Keep edges that some route within budget could traverse (with ``partial``, run part of: for routes that
    may turn around mid-edge once long edges are split)."""
    G = nx.Graph()
    for e in edges:
        if not G.has_edge(e["u"], e["v"]) or G[e["u"]][e["v"]]["w"] > e["length"]:
            G.add_edge(e["u"], e["v"], w=e["length"])
    if not set(starts) & set(G):
        raise VertmaxxerError("The start isn't on any usable trail or road")
    if ends is not None and not set(ends) & set(G):
        raise VertmaxxerError("No end is on a usable trail or road")
    d_s = nx.multi_source_dijkstra_path_length(G, set(starts) & set(G), cutoff=budget, weight="w")
    d_t = d_s if ends is None else nx.multi_source_dijkstra_path_length(G, set(ends) & set(G), cutoff=budget, weight="w")
    inf = math.inf

    def ok(e):
        u, v = e["u"], e["v"]
        if partial:
            return min(d_s.get(u, inf) + d_t.get(u, inf), d_s.get(v, inf) + d_t.get(v, inf)) <= budget
        return min(d_s.get(u, inf) + d_t.get(v, inf), d_s.get(v, inf) + d_t.get(u, inf)) + e["length"] <= budget

    kept = [e for e in edges if ok(e)]
    if not kept:
        raise VertmaxxerError("No route within the distance budget connects the trailhead(s); "
                         "check where they snapped, or try --roads")
    return kept


def _reversed(e):
    return dict(e, u=e["v"], v=e["u"], latlon=e["latlon"][::-1], names=e["names"][::-1], roadpt=e["roadpt"][::-1],
                flatpt=_flatpt(e)[::-1])


def _flatpt(e):
    return e.get("flatpt", np.zeros(len(e["latlon"]), bool))


def contract(edges, keep):
    """Merge chains of edges through nodes of degree 2 (way splits that aren't real junctions)."""
    alive = dict(enumerate(edges))
    next_id = len(edges)
    adj = defaultdict(set)
    for i, e in alive.items():
        adj[e["u"]].add(i)
        adj[e["v"]].add(i)
    for n in list(adj):
        if n in keep or len(adj[n]) != 2:
            continue
        i, j = adj[n]
        if any(alive[k]["u"] == alive[k]["v"] for k in (i, j)):
            continue
        a = alive.pop(i) if alive[i]["v"] == n else _reversed(alive.pop(i))
        b = alive.pop(j) if alive[j]["u"] == n else _reversed(alive.pop(j))
        k, next_id = next_id, next_id + 1
        alive[k] = dict(
            u=a["u"], v=b["v"], latlon=np.vstack([a["latlon"], b["latlon"][1:]]),
            names=np.concatenate([a["names"], b["names"][1:]]),
            roadpt=np.concatenate([a["roadpt"], b["roadpt"][1:]]),
            flatpt=np.concatenate([_flatpt(a), _flatpt(b)[1:]]),
            length=a["length"] + b["length"], road=a["road"] or b["road"],
        )
        del adj[n]
        adj[a["u"]] -= {i, j}
        adj[b["v"]] -= {i, j}
        adj[a["u"]].add(k)
        adj[b["v"]].add(k)
    return list(alive.values())


def access_spur(edges, start, max_m):
    """The access path from ``start`` to the nearest node on a loop of trail (loops closed by roads don't count).

    Returns (edge indices along the path, that node), or ([], start) if ``start`` already lies on such a loop
    or none is within ``max_m``. Routes run this path out (and back) without it counting toward their shape,
    so a trailhead reached by an approach trail can still start a loop.
    """
    # Parallel ways (e.g. a footway mapped alongside a cycleway) collapse to one edge: they aren't a loop.
    G, T = nx.Graph(), nx.Graph()
    for k, e in enumerate(edges):
        if e["u"] == e["v"]:
            continue
        for H, ok in ((G, True), (T, not e["roadpt"].any())):
            if ok and (not H.has_edge(e["u"], e["v"]) or H[e["u"]][e["v"]]["w"] > e["length"]):
                H.add_edge(e["u"], e["v"], w=e["length"], k=k)
    bridges = {frozenset(b) for b in nx.bridges(T)}
    core = {n for u, v in T.edges() if frozenset((u, v)) not in bridges for n in (u, v)}
    core |= {e["u"] for e in edges if e["u"] == e["v"] and not e["roadpt"].any()}  # a loop with one junction
    if start in core or start not in G:
        return [], start
    dist, paths = nx.single_source_dijkstra(G, start, cutoff=max_m, weight="w")
    targets = [n for n in dist if n in core]
    if not targets:
        return [], start
    t = min(targets, key=dist.get)
    return [G[a][b]["k"] for a, b in zip(paths[t][:-1], paths[t][1:])], t


def shortest_loop_through(edges, node):
    """Length (m) of the shortest loop passing through ``node`` (inf if none); parallel ways don't count."""
    G = nx.Graph()
    for e in edges:
        if e["u"] != e["v"] and (not G.has_edge(e["u"], e["v"]) or G[e["u"]][e["v"]]["w"] > e["length"]):
            G.add_edge(e["u"], e["v"], w=e["length"])
    best = math.inf
    for nb in list(G.neighbors(node)) if node in G else []:
        w = G[node][nb]["w"]
        G.remove_edge(node, nb)
        try:
            best = min(best, w + nx.dijkstra_path_length(G, nb, node, weight="w"))
        except nx.NetworkXNoPath:
            pass
        G.add_edge(node, nb, w=w)
    return best


# ---------------------------------------------------------------- elevation

class DEM:
    """Elevation mosaic over a lat/lon box, fetched in tiles and cached.

    ``3dep``: USGS 3DEP ImageServer resampled to 1/3 arc-second (~10 m) from the best available data.
    ``terrarium``: AWS Terrain Tiles at zoom 14 (~7 m in the US, also derived from 3DEP), a fast CDN
    that is useful when the USGS service is down.
    """

    def __init__(self, source, lat_min, lat_max, lon_min, lon_max):
        self.source = source
        T = self.tile_px = DEM_TILE_PX if source == "3dep" else 256
        x0, y0 = self._pixel(lat_max, lon_min)
        x1, y1 = self._pixel(lat_min, lon_max)
        self.i0, self.j0 = int(x0 // T), int(y0 // T)
        tiles = [(i, j) for j in range(self.j0, int(y1 // T) + 1) for i in range(self.i0, int(x1 // T) + 1)]
        missing = sum(not (CACHE_DIR / self._cache_name(i, j)).exists() for i, j in tiles)
        if missing:
            print(f"Fetching {missing} {source} tiles...")
        with ThreadPoolExecutor(4 if source == "3dep" else 8) as pool:
            z = dict(zip(tiles, pool.map(lambda ij: self._tile(*ij), tiles)))
        ni = int(x1 // T) + 1 - self.i0
        self.z = np.vstack([np.hstack([z[tiles[r * ni + c]] for c in range(ni)]) for r in range(len(tiles) // ni)])

    def _pixel(self, lat, lon):
        """Global pixel coordinates (x east, y south) of the source's tile lattice."""
        if self.source == "3dep":
            return lon / DEM_RES_DEG, -lat / DEM_RES_DEG
        n = 256 * 2**TERRARIUM_ZOOM
        return (lon + 180) / 360 * n, (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n

    def _cache_name(self, i, j):
        if self.source == "3dep":
            return f"3dep_{DEM_TILE_PX}_{i}_{j}.npy"
        return f"terrarium_{TERRARIUM_ZOOM}_{i}_{j}.png"

    def _tile(self, i, j):
        if self.source == "3dep":
            d = DEM_TILE_PX * DEM_RES_DEG
            params = dict(
                bbox=f"{i * d:.9f},{-(j + 1) * d:.9f},{(i + 1) * d:.9f},{-j * d:.9f}", bboxSR=4326, imageSR=4326,
                size=f"{DEM_TILE_PX},{DEM_TILE_PX}", format="tiff", pixelType="F32", noData=-9999,
                interpolation="RSP_BilinearInterpolation", f="image",
            )

            def load(p):
                return np.load(p)

            def fetch():
                z = np.array(Image.open(io.BytesIO(_get(DEM_URL, params))), dtype=np.float32)
                z[z < -1000] = np.nan
                return z

            return _cached(self._cache_name(i, j), fetch, load, lambda p, z: np.save(p, z))

        def load(p):
            rgb = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
            return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768

        def fetch():
            return _get(TERRARIUM_URL.format(z=TERRARIUM_ZOOM, x=i, y=j))

        path = CACHE_DIR / self._cache_name(i, j)
        _cached(path.name, fetch, lambda p: None, lambda p, data: p.write_bytes(data))
        return load(path)

    def __call__(self, lat, lon):
        x, y = self._pixel(lat, lon)
        T = self.tile_px
        return map_coordinates(self.z, [y - self.j0 * T - 0.5, x - self.i0 * T - 0.5], order=1, mode="nearest")


def _get(url, params=None):
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException as err:
            if attempt == 3:
                raise VertmaxxerError(f"Elevation request failed: {err}\n"
                                 "The USGS service is intermittently down; try --dem terrarium.")
            time.sleep(5 * 2**attempt)


def _smooth(z, s, length):
    """Forward-backward single-pole IIR along distance ``s`` (any spacing), then pinned to the raw
    endpoint values so that a node's elevation is the same on every edge touching it."""
    if length <= 0 or len(z) < 3:
        return z.copy()
    alpha = 1.0 - np.exp(-np.diff(s) / length)
    y = z.astype(float)
    for i in range(1, len(y)):
        y[i] = alpha[i - 1] * y[i] + (1 - alpha[i - 1]) * y[i - 1]
    for i in range(len(y) - 2, -1, -1):
        y[i] = alpha[i] * y[i] + (1 - alpha[i]) * y[i + 1]
    return y + np.interp(s, [s[0], s[-1]], [z[0] - y[0], z[-1] - y[-1]])


def add_elevation(edges, smoothing_length, source):
    """Resample each edge every ~RESAMPLE_M, keeping its OSM vertices so length is preserved, and attach raw
    and smoothed elevation."""
    ll = np.vstack([e["latlon"] for e in edges])
    dem = DEM(source, ll[:, 0].min() - 0.002, ll[:, 0].max() + 0.002, ll[:, 1].min() - 0.002, ll[:, 1].max() + 0.002)
    for e in edges:
        lat, lon = e["latlon"].T
        s = np.concatenate([[0.0], np.cumsum(_haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
        keep = np.concatenate([[True], np.diff(s) > 0])
        s, lat, lon, names, roadpt = s[keep], lat[keep], lon[keep], e["names"][keep], e["roadpt"][keep]
        flat = _flatpt(e)[keep]
        n = max(1, math.ceil(s[-1] / RESAMPLE_M))
        si = np.union1d(np.linspace(0, s[-1], n + 1), s)
        e["lat"], e["lon"] = np.interp(si, s, lat), np.interp(si, s, lon)
        seg = np.clip(np.searchsorted(s, si, side="right") - 1, 0, len(s) - 2)
        e["names"], e["roadpt"] = names[seg], roadpt[seg]
        e["z_raw"] = _bridge_over(dem(e["lat"], e["lon"]), si, flat[seg])
        e["z"] = _smooth(e["z_raw"], si, smoothing_length)
        e["s"] = si
    if any(np.isnan(e["z_raw"]).any() for e in edges):
        raise VertmaxxerError("DEM has no data along part of the network (outside 3DEP coverage?)")


def _bridge_over(z, s, off_ground):
    """Elevation along bridges and tunnels, interpolated straight between the ground points on either side."""
    z = z.copy()
    i, n = 0, len(z)
    while i < n:
        if not off_ground[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and off_ground[j + 1]:
            j += 1
        a, b = max(i - 1, 0), min(j + 1, n - 1)
        if b > a:
            z[a:b + 1] = np.interp(s[a:b + 1], [s[a], s[b]], [z[a], z[b]])
        i = j + 1
    return z


def subdivide(edges, seg_max):
    """Split edges into pieces of at most ``seg_max`` meters (None: no splitting) and compute each piece's sum |dz|."""
    out = []
    for k, e in enumerate(edges):
        npts = len(e["s"])
        pieces = 1 if seg_max is None else min(npts - 1, max(1, math.ceil(e["length"] / seg_max)))
        cuts = np.round(np.linspace(0, npts - 1, pieces + 1)).astype(int)
        for p in range(pieces):
            i0, i1 = cuts[p], cuts[p + 1]
            sl = slice(i0, i1 + 1)
            out.append(dict(
                u=e["u"] if p == 0 else ("split", k, p),
                v=e["v"] if p == pieces - 1 else ("split", k, p + 1),
                lat=e["lat"][sl], lon=e["lon"][sl], z_raw=e["z_raw"][sl], z=e["z"][sl], names=e["names"][sl],
                roadpt=e["roadpt"][sl],
                length=e["length"] * (e["s"][i1] - e["s"][i0]) / e["s"][-1], road=e["road"],
            ))
    for e in out:
        e["var"] = float(np.sum(np.abs(np.diff(e["z"]))))
        # Grade-adjusted time (in flat-equivalent meters) each way, overall and on road.
        ds = _haversine(e["lat"][:-1], e["lon"][:-1], e["lat"][1:], e["lon"][1:])
        grade = 100 * np.diff(e["z"]) / np.maximum(ds, 1e-6)
        fwd, rev = ds * gap_factor(grade), ds * gap_factor(-grade)
        road = e["roadpt"][:-1]
        e["gap"] = (float(fwd.sum()), float(rev.sum()))
        e["gap_road"] = (float(fwd[road].sum()), float(rev[road].sum()))
        e["road_len"] = float(ds[road].sum())
    return out


# ---------------------------------------------------------------- optimization

def solve(edges, starts, ends, budget, topology, min_loop, time_limit, workers, verbose, hint=None,
          min_loop_frac=0.0, road_time_frac=None, start_cost=None, minimize=False, min_length=0.0,
          no_turnarounds=False, turnaround_ok=(), max_road_frac=None, turn_penalty=0.0):
    """Return (traversal count per edge, start node, end node, proven optimal) for the best route found.

    ``topology`` is a TOPOLOGIES entry. Each loop must be at least ``min_loop`` (m) long and at least
    ``min_loop_frac`` of the route's total distance. With ``road_time_frac``, road segments may take at most
    that share of the route's grade-adjusted time (a segment run once counts both directions' average); with
    ``max_road_frac``, at most that share of its distance. Each turnaround costs ``turn_penalty`` (m of gain), so
    one has to climb at least that much to be worth it.

    An edge's optional ``cost`` (m) replaces its length in the budget, e.g. grade-adjusted time as flat distance.

    ``start_cost`` (m per start) is added to the distance when that start is used, e.g. the walk to it. Edges
    marked ``frozen`` can't be used; an edge with ``link`` = i is run exactly once when start i is used and never
    otherwise (e.g. a walk out via one trailhead and back via another). With ``minimize`` the route climbing the
    least is found instead, at least ``min_length`` (m) long.

    ``hint`` is an optional traversal count per edge (e.g. a route found with a smaller budget) to start from.
    """
    node_ids = sorted({e["u"] for e in edges} | {e["v"] for e in edges}, key=str)
    idx = {n: i for i, n in enumerate(node_ids)}
    N, E = len(node_ids), len(edges)
    eu = [idx[e["u"]] for e in edges]
    ev = [idx[e["v"]] for e in edges]
    L = [round(e["length"]) for e in edges]
    C = [round(e.get("cost", e["length"])) for e in edges]  # what the budget limits: length unless given
    dz = [round(10 * e["var"]) for e in edges]  # decimeters
    z_node = [0] * N
    for e in edges:
        z_node[idx[e["u"]]] = round(10 * e["z"][0])
        z_node[idx[e["v"]]] = round(10 * e["z"][-1])
    p2p = ends is not None
    loops, spurs, start_on = topology["loops"], topology["spurs"], topology["start"]
    reuse, retrace_loops = topology["reuse"], topology.get("retrace_loops", False)
    keep = [i for i, n in enumerate(starts) if n in idx]
    starts = [starts[i] for i in keep]
    if start_cost is not None:
        start_cost = [start_cost[i] for i in keep]
    if not starts:
        raise VertmaxxerError("No start is reachable within the distance budget.")
    S = [idx[n] for n in starts]
    if p2p:
        ends = [n for n in ends if n in idx]
        if not ends:
            raise VertmaxxerError("No end trailhead is reachable within the distance budget.")
    T = [idx[n] for n in ends] if p2p else []

    md = cp_model.CpModel()
    a = [md.NewBoolVar(f"a{e}") for e in range(E)]  # traversed once
    b = [md.NewBoolVar(f"b{e}") for e in range(E)]  # traversed twice
    v = [md.NewBoolVar(f"v{n}") for n in range(N)]
    s = [md.NewBoolVar(f"s{i}") for i in range(len(S))]
    t = [md.NewBoolVar(f"t{j}") for j in range(len(T))]
    md.AddExactlyOne(s)
    for e in range(E):
        if "link" in edges[e]:
            md.Add(a[e] == s[edges[e]["link"]])
    if p2p:
        md.AddExactlyOne(t)

    inc = [[] for _ in range(N)]  # (edge, degree multiplicity)
    for e in range(E):
        md.AddAtMostOne(a[e], b[e])
        if edges[e].get("frozen"):
            md.Add(a[e] + b[e] == 0)
        if edges[e].get("require"):  # traversed exactly this many times (1 or 2)
            md.Add((a[e] if edges[e]["require"] == 1 else b[e]) == 1)
        if edges[e].get("spur_only"):  # only as an out-and-back
            md.Add(a[e] == 0)
        if "link" in edges[e]:
            md.Add(b[e] == 0)
        if not reuse or edges[e].get("once") or (eu[e] == ev[e] and not retrace_loops):
            md.Add(b[e] == 0)
        for n in {eu[e], ev[e]}:
            md.AddImplication(a[e], v[n])
            md.AddImplication(b[e], v[n])
        if eu[e] == ev[e]:
            inc[eu[e]].append((e, 2))
        else:
            inc[eu[e]].append((e, 1))
            inc[ev[e]].append((e, 1))
    # Start/end selectors act as edges to a virtual root: traversed twice for closed routes,
    # once each for point-to-point routes (where the root closes the walk end -> start).
    virt = [[] for _ in range(N)]
    for i, n in enumerate(S):
        virt[n].append(s[i])
        md.AddImplication(s[i], v[n])
    for j, n in enumerate(T):
        virt[n].append(t[j])
        md.AddImplication(t[j], v[n])

    # No turning back except at the start (and turnaround_ok nodes): every other node the route uses touches at
    # least two used edges.
    if no_turnarounds:
        start_lits = {}
        for i, n in enumerate(S):
            start_lits.setdefault(n, []).append(s[i].Not())
        ok = {idx[n] for n in turnaround_ok if n in idx}
        for n in range(N):
            if n in ok:
                continue
            es = {e for e, _ in inc[n] if eu[e] != ev[e]}
            md.Add(sum(a[e] + b[e] for e in es) >= 2).OnlyEnforceIf([v[n]] + start_lits.get(n, []))

    # Connectivity: every used node has exactly one parent, either a used edge to a lower-ranked
    # node or (for the chosen start) the root.
    rank = [md.NewIntVar(1, N, f"r{n}") for n in range(N)]
    parents = [[] for _ in range(N)]
    for e in range(E):
        if eu[e] == ev[e]:
            continue
        for child, par in ((eu[e], ev[e]), (ev[e], eu[e])):
            p = md.NewBoolVar("")
            md.AddBoolOr([a[e], b[e]]).OnlyEnforceIf(p)
            md.Add(rank[child] > rank[par]).OnlyEnforceIf(p)
            parents[child].append(p)
    for i, n in enumerate(S):
        p = md.NewBoolVar("")
        md.AddImplication(p, s[i])
        parents[n].append(p)
    for n in range(N):
        md.Add(sum(parents[n]) == v[n])

    # Retraced edges must form a forest, i.e. admit an orientation along increasing rank with at
    # most one retraced edge entering each node.
    if reuse and not retrace_loops:
        into = [[] for _ in range(N)]
        for e in range(E):
            if eu[e] == ev[e]:
                continue
            q = (md.NewBoolVar(""), md.NewBoolVar(""))
            md.Add(q[0] + q[1] == b[e])
            md.Add(rank[eu[e]] > rank[ev[e]]).OnlyEnforceIf(q[0])
            md.Add(rank[ev[e]] > rank[eu[e]]).OnlyEnforceIf(q[1])
            into[eu[e]].append(q[0])
            into[ev[e]].append(q[1])
        for n in range(N):
            md.AddAtMostOne(into[n])

    total = sum(L[e] * (a[e] + 2 * b[e]) for e in range(E))
    pct = round(100 * min_loop_frac)

    def loop_floor(xs):
        loop = sum(L[e] * xs[e] for e in range(E))
        md.Add(loop >= max(1, int(min_loop)))
        if pct:
            md.Add(100 * loop >= pct * total)

    # Exact loop count. Once-traversed edges form an even-degree subgraph, so any of them implies a
    # loop; the cycle-rank cap below then makes the count exact.
    if loops == 1:
        loop_floor(a)
    elif loops == 2 and not reuse:
        # Figure-8: with nothing retraced, an even-degree connected route of cycle rank 2 can only be two
        # loops sharing one node. Assign each once-traversed edge to loop 0 or 1 with even degree per loop
        # at every node, which splits it at that node, so each loop can be held to the minimum length.
        loop_edges = ([], [])
        for e in range(E):
            x = (md.NewBoolVar(""), md.NewBoolVar(""))
            md.Add(x[0] + x[1] == a[e])
            for k in (0, 1):
                loop_edges[k].append(x[k])
        for n in range(N):
            md.Add(sum(m * loop_edges[1][e] for e, m in inc[n]) == 2 * md.NewIntVar(0, len(inc[n]), ""))
        for xs in loop_edges:
            loop_floor(xs)
    elif loops == 2:
        # Label nodes by side; once-traversed edges stay within a side and each side needs its own
        # loop. With cycle rank <= 2 the loops are then vertex-disjoint and joined by retraced trail.
        side = [md.NewBoolVar(f"side{n}") for n in range(N)]
        loop_edges = ([], [])  # once-traversed edges certified to lie on side 0 / side 1
        for e in range(E):
            md.Add(side[eu[e]] == side[ev[e]]).OnlyEnforceIf(a[e])
            for k, lit in enumerate((side[eu[e]].Not(), side[eu[e]])):
                x = md.NewBoolVar("")
                md.AddImplication(x, a[e])
                md.AddImplication(x, lit)
                loop_edges[k].append(x)
        for xs in loop_edges:
            loop_floor(xs)
        if start_on == "loop":
            for i, n in enumerate(S):
                md.AddImplication(s[i], side[n].Not())
    for i, n in enumerate(S):
        deg_a = sum(m * a[e] for e, m in inc[n])
        if start_on == "loop":
            md.Add(deg_a >= 2).OnlyEnforceIf(s[i])
        elif start_on == "stem":
            md.Add(deg_a == 0).OnlyEnforceIf(s[i])

    leaves = []
    for n in range(N):
        deg_a = sum(m * a[e] for e, m in inc[n]) + (sum(virt[n]) if p2p else 0)
        md.Add(deg_a == 2 * md.NewIntVar(0, len(inc[n]) + 1, f"k{n}"))
        if spurs is not None or turn_penalty:
            leaf = md.NewBoolVar(f"leaf{n}")
            md.Add(sum(m * (a[e] + b[e]) for e, m in inc[n]) + sum(virt[n]) + leaf >= 2 * v[n])
            leaves.append(leaf)
    extra = sum(round(c) * s[i] for i, c in enumerate(start_cost)) if start_cost is not None else 0
    md.Add(sum(C[e] * (a[e] + 2 * b[e]) for e in range(E)) + extra <= int(budget))
    if loops is not None:
        md.Add(sum(a) + sum(b) - sum(v) <= loops - 1)
    if road_time_frac is not None:
        t1 = [round(sum(e["gap"]) / 2) for e in edges]
        r1 = [round(sum(e["gap_road"]) / 2) for e in edges]
        pct = round(100 * road_time_frac)
        md.Add(100 * sum(r1[e] * (a[e] + 2 * b[e]) for e in range(E))
               <= pct * sum(t1[e] * (a[e] + 2 * b[e]) for e in range(E)))
    if max_road_frac is not None:
        rl = [round(e["road_len"]) for e in edges]
        pct = round(100 * max_road_frac)
        md.Add(100 * sum(rl[e] * (a[e] + 2 * b[e]) for e in range(E))
               <= pct * sum(L[e] * (a[e] + 2 * b[e]) for e in range(E)))
    if spurs is not None:
        md.Add(sum(leaves) <= spurs)

    # Objective in units of 2 * gain in decimeters.
    objective = sum(dz[e] * (a[e] + 2 * b[e]) for e in range(E))
    if p2p:
        objective += sum(z_node[n] * t[j] for j, n in enumerate(T)) - sum(z_node[n] * s[i] for i, n in enumerate(S))
    if turn_penalty:
        objective += (1 if minimize else -1) * round(20 * turn_penalty) * sum(leaves)
    if min_length:
        md.Add(sum(L[e] * (a[e] + 2 * b[e]) for e in range(E)) + extra >= int(min_length))
    if minimize:
        md.Minimize(objective)
    else:
        md.Maximize(objective)
    if hint is not None:
        for e in range(E):
            md.AddHint(a[e], hint[e] == 1)
            md.AddHint(b[e], hint[e] == 2)
        # A hint on the edges alone is often not completed in time on large graphs. Solve for the remaining
        # variables (ranks, orientations, sides) with the edges fixed, and hint the full assignment.
        fixed = md.clone()
        for e in range(E):
            for x, val in ((a[e], int(hint[e] == 1)), (b[e], int(hint[e] == 2))):
                dom = fixed.proto.variables[x.Index()].domain
                dom[0], dom[1] = val, val
        fs = cp_model.CpSolver()
        fs.parameters.max_time_in_seconds = min(120.0, time_limit / 4)
        fs.parameters.num_workers = workers
        if fs.Solve(fixed) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            md.ClearHints()
            for i, val in enumerate(fs.response_proto.solution):
                md.AddHint(md.get_int_var_from_proto_index(i), val)
            print(f"  hint completed: gain {fs.ObjectiveValue() / 20 * M_TO_FT:,.0f} ft")
        else:
            print("  hint could not be completed; using it on the edges only")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = verbose
    t0 = time.time()

    turn_units = round(20 * turn_penalty)

    def gain_ft(value, sol):
        """The route's gain from the objective, without the turnaround costs it includes."""
        if turn_units:
            value += (-1 if minimize else 1) * turn_units * sum(sol.Value(x) for x in leaves)
        return value / 20 * M_TO_FT

    class Progress(cp_model.CpSolverSolutionCallback):
        best = None

        def on_solution_callback(self):
            gain = gain_ft(self.ObjectiveValue(), self)
            if _improved(gain, self.best, minimize):
                score = f"score {self.ObjectiveValue() / 20 * M_TO_FT:,.0f} ft, " if turn_units else ""
                print(f"  {time.time() - t0:6.1f} s  gain {gain:7,.0f} ft  "
                      f"({score}bound {self.BestObjectiveBound() / 20 * M_TO_FT:,.0f} ft)")
                self.best = gain

    print(f"Solving: {N} nodes, {E} edges, {time_limit:.0f} s limit, {workers} workers")
    status = solver.Solve(md, Progress())
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise VertmaxxerError(_no_route(solver.StatusName(status), time_limit))
    score = (f"score {solver.ObjectiveValue() / 20 * M_TO_FT:,.0f} ft (gain less the turnaround costs), "
             if turn_units else "")
    print(f"  {solver.StatusName(status).lower()}: gain {gain_ft(solver.ObjectiveValue(), solver):,.0f} ft, "
          f"{score}bound {solver.BestObjectiveBound() / 20 * M_TO_FT:,.0f} ft")

    m = np.array([solver.Value(a[e]) + 2 * solver.Value(b[e]) for e in range(E)])
    start = starts[next(i for i in range(len(S)) if solver.Value(s[i]))]
    end = ends[next(j for j in range(len(T)) if solver.Value(t[j]))] if p2p else start
    return m, start, end, status == cp_model.OPTIMAL


# ---------------------------------------------------------------- output

def _improved(gain, best, minimize):
    """Whether a solution is worth a progress line: 0.5% better than the last one shown."""
    return best is None or (gain < 0.995 * best if minimize else gain > 1.005 * best)


def _no_route(status, time_limit):
    if status == "UNKNOWN":
        return (f"No route found within the {time_limit:.0f} s time limit; try a longer --time-limit "
                "(or more --workers)")
    return f"No feasible route found ({status})"


def assemble(edges, m, start, end):
    G = nx.MultiGraph()
    for e in np.flatnonzero(m):
        for _ in range(m[e]):
            G.add_edge(edges[e]["u"], edges[e]["v"], eid=e)
    if G.number_of_edges() == 0:
        raise VertmaxxerError("Empty route: no edge fits within the budget.")
    walk = nx.eulerian_path(G, source=start, keys=True) if end != start else nx.eulerian_circuit(G, source=start, keys=True)
    parts = []
    for u, v, k in walk:
        e = edges[G[u][v][k]["eid"]]
        sl = slice(None) if e["u"] == u else slice(None, None, -1)
        parts.append({key: e[key][sl] for key in ("lat", "lon", "z_raw", "z", "names")})
    route = {key: np.concatenate([parts[0][key]] + [p[key][1:] for p in parts[1:]])
             for key in ("lat", "lon", "z_raw", "z", "names")}
    step = _haversine(route["lat"][:-1], route["lon"][:-1], route["lat"][1:], route["lon"][1:])
    route["dist"] = np.concatenate([[0.0], np.cumsum(step)])
    legs = []
    for name, d in zip(route["names"][1:], step):
        if legs and legs[-1][0] == name:
            legs[-1][1] += d
        else:
            legs.append([name, d])
    route["legs"] = legs
    return route


def classify(edges, m, start, end, min_loop, min_loop_frac=0.0):
    """Name a route's shape from its traversal counts alone. Loops shorter than ``min_loop`` (m) or than
    ``min_loop_frac`` of the route don't count as loops, so such routes are "other". Returns (shape, details)."""
    min_loop = max(min_loop, min_loop_frac * sum(edges[i]["length"] * m[i] for i in np.flatnonzero(m)))
    once, twice, foot = nx.MultiGraph(), nx.MultiGraph(), nx.MultiGraph()
    for i in np.flatnonzero(m):
        e = edges[i]
        (once if m[i] == 1 else twice).add_edge(e["u"], e["v"], length=e["length"])
        foot.add_edge(e["u"], e["v"])

    def miles(G):
        return sum(d["length"] for *_, d in G.edges(data=True)) / MI_TO_M

    rank = foot.number_of_edges() - foot.number_of_nodes() + 1
    if end != start:
        if rank == 0 and twice.number_of_edges() == 0:
            return "traverse", f"{miles(once):.2f} mi"
        return "other", f"point-to-point with {rank} loops and {miles(twice):.2f} mi retraced"
    if (rank == 2 and twice.number_of_edges() == 0 and nx.is_connected(once) and once.has_node(start)
            and sorted(d for _, d in once.degree())[-2:] == [2, 4]):
        hub = next(n for n, d in once.degree() if d == 4)
        rest = once.copy()
        rest.remove_node(hub)
        halves = [sum(d["length"] for u, v, d in once.edges(data=True) if u in c or v in c) / MI_TO_M
                  for c in nx.connected_components(rest)]
        halves += [d["length"] / MI_TO_M for u, v, d in once.edges(hub, data=True) if u == v]  # one-edge loops
        if len(halves) == 2 and min(halves) * MI_TO_M >= 0.995 * min_loop:
            return "figure-8", "loops " + " + ".join(f"{h:.2f}" for h in sorted(halves, reverse=True)) + " mi"
    loops = [once.subgraph(c) for c in nx.connected_components(once)]
    simple = all(g.number_of_edges() == g.number_of_nodes() and all(d == 2 for _, d in g.degree()) for g in loops)
    turnarounds = sum(d == 1 and n != start for n, d in foot.degree())
    retraced = f"{miles(twice):.2f} mi"
    loop_mi = " + ".join(f"{miles(g):.2f}" for g in sorted(loops, key=lambda g: start not in g)) + " mi"
    short = [g for g in loops if miles(g) * MI_TO_M < 0.995 * min_loop]  # slack for the solver's whole-meter lengths
    if short:
        return "other", f"{len(short)} loop(s) under the minimum ({loop_mi}), {retraced} retraced"
    if simple and rank == len(loops):  # retraced trail only connects, never closes a cycle of its own
        on_loop = once.has_node(start)
        if not loops and turnarounds == 1:
            return "out-and-back", f"{retraced} each way"
        if turnarounds == 0 and len(loops) == 1:
            return ("loop", loop_mi) if on_loop else ("lollipop", f"stem {retraced}, loop {loop_mi}")
        if turnarounds and len(loops) == 1:
            return "loop-spurs", f"loop {loop_mi}, {turnarounds} out-and-back spur(s), {retraced} retraced"
        if turnarounds == 0 and len(loops) == 2:
            if on_loop:
                return "dumbbell", f"bridge {retraced}, loops {loop_mi} (starting loop first)"
            return "double-lollipop", f"stem and bridge {retraced}, loops {loop_mi}"
    return "other", f"{rank} loops, {turnarounds} turnarounds, {retraced} retraced"


def write_gpx(path, route, name):
    name = xml_escape(str(name))
    pts = "\n".join(f'<trkpt lat="{la:.7f}" lon="{lo:.7f}"><ele>{z:.1f}</ele></trkpt>'
                    for la, lo, z in zip(route["lat"], route["lon"], route["z_raw"]))
    Path(path).write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="vertmaxxer" xmlns="http://www.topografix.com/GPX/1/1">\n'
        f"<trk><name>{name}</name><trkseg>\n{pts}\n</trkseg></trk>\n</gpx>\n"
    )


def plot(path, edges, route, anchors):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, (ax_map, ax_prof) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    for e in edges:
        ax_map.plot(e["lon"], e["lat"], color="tan" if e["road"] else "0.7", lw=0.6, zorder=1)
    sc = ax_map.scatter(route["lon"], route["lat"], c=route["z_raw"] * M_TO_FT, s=2, cmap="viridis", zorder=2)
    ax_map.plot(*np.array(anchors)[:, ::-1].T, "r*", ms=12, zorder=3)
    ax_map.set_aspect(1 / math.cos(math.radians(np.mean(route["lat"]))))
    fig.colorbar(sc, ax=ax_map, label="elevation (ft)", shrink=0.8)
    ax_prof.plot(route["dist"] / MI_TO_M, route["z_raw"] * M_TO_FT, lw=0.8, label="3DEP")
    ax_prof.plot(route["dist"] / MI_TO_M, route["z"] * M_TO_FT, lw=1.2, label="smoothed")
    ax_prof.set_xlabel("distance (mi)")
    ax_prof.set_ylabel("elevation (ft)")
    ax_prof.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------------------------------------------------------------- CLI
