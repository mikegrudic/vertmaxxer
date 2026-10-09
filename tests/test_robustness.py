"""find_route, spurify and the CLIs on hand-built OSM data with a fake elevation model (no network)."""
import json
import os
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

import vertmaxxer as vm
from vertmaxxer import api, cli, core

M_PER_DEG = 111195.0


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    """Each test with its own (empty) saved settings, and imperial output."""
    monkeypatch.setenv("VERTMAXXER_CONFIG", str(tmp_path / "settings.json"))
    monkeypatch.setattr(core, "METRIC", False)


class FakeDEM:  # elevation rises 50 m per km northward
    def __init__(self, *a, **k):
        pass

    def __call__(self, lat, lon):
        return (np.asarray(lat) - 44.0) * M_PER_DEG * 0.05


def osm(ways):
    """ways: [(way_id, [(lat, lon), ...], tags)] -> Overpass-style JSON; shared coordinates share a node."""
    els, nid = [], {}
    for wid, pts, tags in ways:
        ids = []
        for la, lo in pts:
            key = (round(la, 7), round(lo, 7))
            if key not in nid:
                nid[key] = len(nid) + 1
                els.append(dict(type="node", id=nid[key], lat=key[0], lon=key[1]))
            ids.append(nid[key])
        els.append(dict(type="way", id=wid, nodes=ids, tags=tags))
    return dict(elements=els)


def line(lat0, lon0, km, step_m=50, south=False):
    return [(lat0 + (-1 if south else 1) * i * step_m / M_PER_DEG, lon0) for i in range(int(km * 1000 / step_m) + 1)]


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Serve ``data`` as the OSM download, with the fake DEM and a private cache."""
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(core, "DEM", FakeDEM)

    def serve(data):
        monkeypatch.setattr(core, "fetch_osm", lambda *a, **k: data)
    return serve


def run(start, *a, **k):
    k.setdefault("time_limit_s", 10)
    k.setdefault("workers", 2)
    return vm.find_route(start, *a, **k)


PATH = {"highway": "path", "name": "Long Trail"}
PTS = line(44.0, -72.0, 5)


@pytest.mark.parametrize("ways", [
    [(1, PTS, PATH)],
    [(1, PTS[:21], PATH), (2, PTS[20:], PATH)],
    [(i + 1, PTS[i * 10:(i + 1) * 10 + 1], PATH) for i in range(10)],
])
def test_out_and_back_turns_around_mid_way(offline, ways):
    """However OSM splits the trail into ways, a 4 mi out-and-back runs about 2 mi out and back."""
    offline(osm(ways))
    r = run(PTS[0], 4, "out-and-back")
    assert r.distance_mi > 3.5


def tri_and_approach():
    A = (44.0, -72.0)
    B = (A[0] + 1000 / M_PER_DEG, A[1])
    C = (A[0] + 500 / M_PER_DEG, A[1] + 1000 / (M_PER_DEG * np.cos(np.radians(44))))
    seg = lambda p, q: [(p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])) for f in np.linspace(0, 1, 21)]
    tri = seg(A, B) + seg(B, C)[1:] + seg(C, A)[1:]
    approach = line(*A, 5, south=True)
    return A, approach, osm([(1, tri, {"highway": "path", "name": "Triangle"}),
                             (2, approach, {"highway": "path", "name": "Approach"})])


def test_unreachable_extra_start_is_ignored(offline):
    A, approach, net = tri_and_approach()
    offline(net)
    assert run([A, approach[-1]], 2.5, "loop").shape == "loop"


def grid(with_streets=True):
    hw = [(44.0, -72.0 + i * 0.001) for i in range(-10, 11)]
    st = lambda lon: [(44.0 + j * 0.001, lon) for j in range(-10, 11)]
    ways = [(100, hw, {"highway": "primary", "name": "US 1"})]
    if with_streets:
        ways += [(101, st(-72.005), {"highway": "residential", "name": "West St"}),
                 (102, st(-71.995), {"highway": "residential", "name": "East St"}),
                 (103, [(44.008, -72.005), (44.008, -71.995)], {"highway": "residential", "name": "North St"}),
                 (104, [(43.992, -72.005), (43.992, -71.995)], {"highway": "residential", "name": "South St"})]
    return osm(ways)


def test_roads_only_start_on_a_highway_uses_the_streets(offline):
    offline(grid())
    assert run((44.0001, -72.0), 4, "loop", roads_only=True).shape == "loop"


def test_no_trails_or_streets_is_a_plain_error(offline):
    offline(grid(with_streets=False))
    with pytest.raises(vm.VertmaxxerError):
        run((44.0, -72.0), 2, "loop")


def test_no_qualifying_end_trailhead_is_a_plain_error(offline, monkeypatch):
    offline(osm([(1, PTS, PATH)]))
    monkeypatch.setattr(core, "road_trailheads", lambda *a, **k: {})
    with pytest.raises(vm.VertmaxxerError, match="trailhead"):
        run(PTS[0], 3, "traverse", end_trailheads=True)


def test_included_ways_bypass_tag_filters():
    sidewalk = {"highway": "footway", "footway": "sidewalk"}
    data = osm([(1, PTS[:5], PATH), (2, PTS[4:9], sidewalk)])
    ways = {e["way"] for e in core.build_graph(data, True, None, [PTS[0]])[0]}
    assert 2 not in ways
    ways = {e["way"] for e in core.build_graph(data, True, None, [PTS[0]], include={2})[0]}
    assert 2 in ways


def test_timeout_is_not_reported_as_infeasible():
    assert "time limit" in core._no_route("UNKNOWN", 120)
    assert "time limit" not in core._no_route("INFEASIBLE", 120)


# ---------------------------------------------------------------- downloads

def test_truncated_cache_file_is_refetched(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    (tmp_path / "osm_x.json").write_text('{"elements": [{"ty')
    load, save = (lambda p: json.loads(p.read_text())), (lambda p, d: p.write_text(json.dumps(d)))
    assert core._cached("osm_x.json", lambda: {"elements": []}, load, save) == {"elements": []}
    assert json.loads((tmp_path / "osm_x.json").read_text()) == {"elements": []}


def test_failed_cache_write_leaves_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)

    def save(p, d):
        p.write_text('{"elem')
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        core._cached("osm_y.json", lambda: {}, json.loads, save)
    assert not (tmp_path / "osm_y.json").exists()


class Reply:
    def __init__(self, data, status=200):
        self.data, self.status_code, self.text = data, status, json.dumps(data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise core.requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self.data


def test_overpass_runtime_error_is_retried_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    replies = [Reply({"elements": [], "remark": "runtime error: Query timed out in \"query\" at line 1"}),
               Reply({"elements": [{"type": "node", "id": 1, "lat": 0, "lon": 0}]})]
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: replies.pop(0))
    assert len(core._overpass("q", "osm")["elements"]) == 1


def test_overpass_keeps_trying_for_minutes(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    slept = []
    monkeypatch.setattr(core.time, "sleep", slept.append)
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: Reply({}, 504))
    with pytest.raises(vm.VertmaxxerError):
        core._overpass("q", "osm")
    assert sum(slept) >= 240


# ---------------------------------------------------------------- spurify

def spur_network():
    """A loop of zigzag trail (so a straight track reads 3% short) with two side trails up to named peaks."""
    def zigzag(p, q, n=40, amp_m=7.0):
        t = np.linspace(0, 1, n + 1)
        la, lo = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
        off = amp_m / M_PER_DEG * np.where(np.arange(n + 1) % 2, 1, -1) * (np.arange(n + 1) % (n) != 0)
        return list(zip(la, lo + off / np.cos(np.radians(44)))), list(zip(la, lo))
    corners = [(44.0, -72.0), (44.02, -72.0), (44.02, -71.97), (44.0, -71.97), (44.0, -72.0)]
    trail, track = [], []
    for p, q in zip(corners, corners[1:]):
        z, s = zigzag(p, q)
        trail += z if not trail else z[1:]
        track += s if not track else s[1:]
    peaks = [(44.014, -71.996), (44.026, -71.985)]  # both up the fake slope (it rises northward)
    west, north = trail[20], trail[60]  # trail nodes mid-way along the west and north sides
    ways = [(1, trail, PATH),
            (2, [west, peaks[0]], {"highway": "path", "name": "West Spur"}),
            (3, [north, peaks[1]], {"highway": "path", "name": "North Spur"}),
            (4, [trail[30], trail[31]], {"highway": "footway"})]  # beside the trail: a ~55 m parallel way
    return osm(ways), np.array(track), peaks


@pytest.fixture
def spur_offline(offline, monkeypatch):
    data, track, peaks = spur_network()
    offline(data)
    monkeypatch.setattr(core, "_overpass", lambda q, prefix: dict(elements=[
        dict(type="node", id=1000 + i, lat=la, lon=lo, tags=dict(name=f"Peak {i}")) for i, (la, lo) in enumerate(peaks)]))
    return track


def test_spurify_matches_the_whole_loop(spur_offline):
    """Three junctions on a long loop: the matched route must go all the way round, not back the short way."""
    s = vm.spurify(spur_offline, extra_mi=0.05, time_limit_s=10, workers=2)
    track_mi = api._along(spur_offline)[-1] / core.MI_TO_M
    assert track_mi < s.base_distance_mi < 1.05 * track_mi  # the zigzag trail is a little longer than the track


def test_spurify_budget_counts_from_the_matched_trail(spur_offline):
    s = vm.spurify(spur_offline, extra_mi=0.05, time_limit_s=10, workers=2)
    assert s.route.distance_mi <= s.base_distance_mi + 0.05 + 1e-6


def test_spurify_side_trips_in_route_order(spur_offline, capsys):
    s = vm.spurify(spur_offline, extra_mi=3, time_limit_s=10, workers=2)
    miles = [t.leaves_at_mi for t in s.side_trips]
    assert len(miles) == 2 and miles == sorted(miles)  # the two summits; nothing up and down the parallel way
    assert capsys.readouterr().out.count("snapped") == 1  # the closed route's start, not every peak


def test_spurify_budget_below_the_route_is_a_plain_error(spur_offline):
    with pytest.raises(vm.VertmaxxerError, match="shorter"):
        vm.spurify(spur_offline, budget_mi=1, time_limit_s=10, workers=2)


def test_spurify_empty_gpx(tmp_path):
    p = tmp_path / "empty.gpx"
    p.write_text("<gpx></gpx>")
    with pytest.raises(vm.VertmaxxerError):
        vm.spurify(str(p), extra_mi=1)


# ---------------------------------------------------------------- options

@pytest.mark.parametrize("kw", [dict(distance_mi=0), dict(distance_mi=-3), dict(time_h=1, pace=0),
                                dict(time_h=1, gap=0), dict(time_h=0, gap=9)])
def test_nonpositive_budgets(kw):
    with pytest.raises(vm.OptionError):
        vm.find_route((44.0, -72.0), **kw)


def test_missing_files_are_option_errors():
    with pytest.raises(vm.OptionError):
        vm.find_route((44.0, -72.0), 5, ways="no/such/file.json")
    with pytest.raises(vm.OptionError):
        vm.find_route((44.0, -72.0), 5, closures=["no/such/file.json"])


def test_paths_as_str_or_pathlike(offline, tmp_path):
    offline(osm([(1, PTS, PATH)]))
    c = tmp_path / "c.json"
    c.write_text(json.dumps({"closed_segments": []}))
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"exclude": [999]}))
    assert run(PTS[0], 2, "out-and-back", closures=str(c), ways=w).shape == "out-and-back"


def test_cli_coordinates():
    assert cli._latlon("41.42698, -73.96568") == (41.42698, -73.96568)
    with pytest.raises(cli.argparse.ArgumentTypeError, match="quote"):
        cli._latlon("41.42698,")


def test_cli_time_needs_a_clock():
    assert cli._time_h("1:30") == 1.5 and cli._time_h("2") == 2
    with pytest.raises(cli.argparse.ArgumentTypeError):
        cli._time_h("90")


def test_cli_usage_errors_only_for_options(monkeypatch):
    def boom(*a, **k):
        raise ValueError("internal")
    monkeypatch.setattr(cli, "find_route", boom)
    with pytest.raises(ValueError, match="internal"):
        cli.main(["--start", "44,-72", "--distance", "5"])


def test_cli_without_cpu_count(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0


def test_spurify_cli_takes_dem(monkeypatch):
    got = {}

    def fake(*a, **k):
        got.update(k)
        raise vm.VertmaxxerError("stop")
    monkeypatch.setattr(cli, "spurify", fake)
    with pytest.raises(SystemExit):
        cli.spurify_main(["x.gpx", "--extra", "1", "--dem", "terrarium"])
    assert got["dem"] == "terrarium"


def test_progress_is_line_buffered_when_piped():
    code = "import sys; from vertmaxxer import cli; cli._line_buffered(); print(sys.stdout.line_buffering)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    assert out.strip() == "True"


def tips(r, max_m=100):
    """Where the route turns back on itself within ``max_m`` of a point it then retraces: short spurs."""
    p = np.c_[r.lat, r.lon]
    d = np.r_[0, np.cumsum(core._haversine(p[:-1, 0], p[:-1, 1], p[1:, 0], p[1:, 1]))]
    out = []
    for i in range(1, len(p) - 1):
        k = 0
        while i - k - 1 >= 0 and i + k + 1 < len(p) and core._haversine(*p[i - k - 1], *p[i + k + 1]) < 1:
            k += 1
        if k and d[i] - d[i - k] < max_m:
            out.append((round(p[i, 0], 5), round(p[i, 1], 5)))
    return out


def test_no_short_road_spurs(offline):
    """A block of streets with a 40 m dead-end stub and a cross street 10 m off the main one: with budget to spare,
    the route doesn't nip a few meters up either for their little climb."""
    north = lambda la, lo, m: (la + m / M_PER_DEG, lo)
    lon_w, lon_e, lon_m = -72.005, -71.995, -72.0
    S, N = 44.0, 44.008
    st = lambda la, lo0, lo1: [(la, lo0 + f * (lo1 - lo0)) for f in np.linspace(0, 1, 11)]
    ns = lambda lo, la0, la1: [(la0 + f * (la1 - la0), lo) for f in np.linspace(0, 1, 11)]
    y = north(S, lon_m, 10)
    data = osm([(1, st(S, lon_w, lon_e), {"highway": "residential", "name": "South St"}),
                (2, st(N, lon_w, lon_e), {"highway": "residential", "name": "North St"}),
                (3, ns(lon_w, S, N), {"highway": "residential", "name": "West St"}),
                (4, ns(lon_e, S, N), {"highway": "residential", "name": "East St"}),
                (5, [(S, lon_m), y], {"highway": "residential", "name": "Mid St"}),
                (6, [(y[0], lon_m - 0.002), y, (y[0], lon_m + 0.002)], {"highway": "residential", "name": "Cross St"}),
                (7, [(S, -71.998), north(S, -71.998, 40)], {"highway": "service", "name": "Stub"}),
                # a 60 m street ending where two 30 m stubs fork off: once they go, it's a short dead end too
                (8, [(S, -71.997), north(S, -71.997, 60)], {"highway": "residential", "name": "Short St"}),
                (9, [north(S, -71.997, 60), north(S, -71.9972, 85)], {"highway": "service"}),
                (10, [north(S, -71.997, 60), north(S, -71.9968, 85)], {"highway": "service"}),
                # two footways joining the same two points 12 m apart: a mini-loop
                (11, [(S, -72.003), north(S, -72.003, 12)], {"highway": "footway"}),
                (12, [(S, -72.003), north(S, -72.003, 6), north(S, -72.003, 12)], {"highway": "footway"})])
    offline(data)
    r = run((S, lon_w), 3.3, "any", roads=True)
    assert r.distance_mi > 2.5 and tips(r) == []


def test_parallel_short_ways_collapse():
    e = lambda u, v, L, road=False: dict(u=u, v=v, length=L, road=road)
    edges = api._tidy([e(1, 2, 3), e(2, 1, 12), e(2, 3, 400), e(3, 2, 900), e(3, 4, 50, True)], {1})
    assert sorted((x["u"], x["v"], x["length"]) for x in edges) == [(1, 2, 3), (2, 3, 400), (3, 2, 900)]


# ---------------------------------------------------------------- follow-up report

def test_lollipop_seed_is_a_stem_and_a_loop():
    e = lambda u, v, L, var, road=0: dict(u=u, v=v, length=L, var=var, road_len=road)
    edges = [e("S", "A", 1000, 50), e("A", "B", 1000, 80), e("B", "C", 1000, 80), e("C", "A", 1000, 80)]
    counts = api._seed(edges, "S", 6000, core.TOPOLOGIES["lollipop"], 1609, 0.25)
    assert list(counts) == [2, 1, 1, 1]
    assert api._seed(edges, "A", 6000, core.TOPOLOGIES["loop"], 1609, 0.25).tolist() == [0, 1, 1, 1]
    assert api._seed(edges, "S", 4000, core.TOPOLOGIES["lollipop"], 1609, 0.25) is None  # 2 + 3 km doesn't fit
    # With road walking capped, a stem that walks a road doesn't count.
    edges[0]["road_len"] = 1000
    assert api._seed(edges, "S", 6000, core.TOPOLOGIES["lollipop"], 1609, 0.25, max_road_frac=0.1) is None
    assert list(api._seed(edges, "S", 6000, core.TOPOLOGIES["lollipop"], 1609, 0.25)) == [2, 1, 1, 1]


def test_minimize_progress_reports_improvements():
    assert core._improved(100, None, True) and core._improved(90, 100, True) and not core._improved(110, 100, True)
    assert core._improved(110, 100, False) and not core._improved(90, 100, False)


def test_progress_gain_is_the_routes(offline, capsys):
    trail = [(i + 1, PTS[i * 10:(i + 1) * 10 + 1], PATH) for i in range(10)]
    offline(osm(trail))
    r = run(PTS[0], 4, "out-and-back")
    final = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip().startswith(("optimal:", "feasible:"))][-1]
    shown = float(final.split("gain")[1].split("ft")[0].replace(",", ""))
    assert abs(shown - r.gain_ft) < 3


def test_included_way_without_highway_tag(offline):
    pier = (77, [PTS[0], (43.999, -72.0), (43.999, -72.001)], {"man_made": "pier"})
    offline(osm([(1, PTS, PATH), pier]))
    r = run(PTS[0], 2, "out-and-back", ways={"include": [77]})
    assert r.shape == "out-and-back"


def test_malformed_files_are_option_errors(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"include": [1, 2,')
    with pytest.raises(vm.OptionError, match="bad.json"):
        vm.find_route((44.0, -72.0), 5, ways=bad)
    notc = tmp_path / "notc.json"
    notc.write_text('{"include": [1]}')
    with pytest.raises(vm.OptionError, match="notc.json"):
        vm.find_route((44.0, -72.0), 5, closures=notc)


def test_overpass_out_of_memory_stops_at_once(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    slept = []
    monkeypatch.setattr(core.time, "sleep", slept.append)
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: Reply(
        {"elements": [], "remark": "runtime error: Query run out of memory using about 2048 MB of RAM."}))
    with pytest.raises(vm.VertmaxxerError, match="shorter distance"):
        core._overpass("q", "osm")
    assert not slept


def test_overpass_timeouts_stop_after_a_few(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: calls.append(1) or Reply(
        {"elements": [], "remark": "runtime error: Query timed out in \"query\" at line 1 after 301 seconds."}))
    with pytest.raises(vm.VertmaxxerError, match="timing out"):
        core._overpass("q", "osm")
    assert len(calls) == 3


def test_dead_mirror_is_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(core, "_DEAD_MIRRORS", set())
    monkeypatch.setattr(core.time, "sleep", lambda s: None)
    urls = []

    def post(url, *a, **k):
        urls.append(url)
        return Reply({}, 504 if url == core.OVERPASS_URLS[0] else 500)
    monkeypatch.setattr(core.requests, "post", post)
    with pytest.raises(vm.VertmaxxerError):
        core._overpass("q", "osm")
    for mirror in core.OVERPASS_URLS[1:]:
        assert urls.count(mirror) == 1


def test_manitou_campus_is_closed():
    closed = core.load_closures(sorted(core.CLOSURES_DIR.glob("*.json")))
    campus = json.load(open(core.CLOSURES_DIR / "manitou_school_2026-10.json"))["closed_segments"]
    assert len(campus) >= 20 and all(frozenset(s) in closed for s in campus)


def test_moffatt_drive_is_closed():
    closed = core.load_closures(sorted(core.CLOSURES_DIR.glob("*.json")))
    drive = json.load(open(core.CLOSURES_DIR / "moffatt_healy_drive_2026-10.json"))["closed_segments"]
    assert drive and all(frozenset(s) in closed for s in drive)


# ---------------------------------------------------------------- adversarial round 1

def test_overpass_refusals(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(core, "_DEAD_MIRRORS", set())
    slept, urls = [], []
    monkeypatch.setattr(core.time, "sleep", slept.append)

    def post(url, *a, **k):
        urls.append(url)
        return Reply({}, 504 if url == core.OVERPASS_URLS[0] else 403)
    monkeypatch.setattr(core.requests, "post", post)
    with pytest.raises(vm.VertmaxxerError):
        core._overpass("q1", "osm")
    assert urls.count(core.OVERPASS_URLS[1]) == 1  # a mirror that refuses us isn't asked again
    slept.clear()
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: Reply({}, 400))
    with pytest.raises(vm.VertmaxxerError, match="rejected"):
        core._overpass("q2", "osm")
    assert not slept


def test_cached_download_covering_a_smaller_query_is_reused(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    data = osm([(1, PTS, PATH)])
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: Reply(data))
    assert core.fetch_osm([(44.0, -72.0)], 5000, True) == data

    def no_network(*a, **k):
        raise AssertionError("downloaded again")
    monkeypatch.setattr(core.requests, "post", no_network)
    assert core.fetch_osm([(44.001, -72.0)], 3000, True) == data  # inside the first circle
    assert core.fetch_osm([(44.0, -72.0)], 5000, False) == data  # trails only: the download had them
    assert core.fetch_osm([(44.003, -72.0)], 5000, True) == data  # nudged, within the download's 1 km margin
    with pytest.raises(AssertionError):
        core.fetch_osm([(44.0, -72.0)], 7000, True)  # bigger: needs a new download


def test_empty_overpass_answers_are_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: Reply({"elements": []}))
    core._overpass("q", "osm")
    assert not list(tmp_path.glob("osm_*"))


def test_end_at_the_start_is_an_option_error():
    with pytest.raises(vm.OptionError, match="loop"):
        vm.find_route((44.0, -72.0), 5, "traverse", end=(44.0005, -72.0))


def test_end_on_a_street_snaps_to_the_street(offline, capsys):
    street = [(PTS[-1][0], -72.003), PTS[-1], (PTS[-1][0], -71.9985), (PTS[-1][0], -71.997)]  # from the trail's end
    offline(osm([(1, PTS, PATH), (2, street, {"highway": "residential", "name": "North St"})]))
    run(PTS[0], 4, "traverse", end=(PTS[-1][0], -71.9985))
    assert "(North St)" in capsys.readouterr().out


def test_far_points_warn_or_fail(offline, capsys):
    offline(osm([(1, PTS, PATH)]))
    run((PTS[0][0] - 300 / M_PER_DEG, -72.0), 4, "out-and-back")
    assert "m from the nearest usable way" in capsys.readouterr().err
    with pytest.raises(vm.VertmaxxerError, match="within 1 km"):
        run((PTS[0][0] - 2000 / M_PER_DEG, -72.0), 4, "out-and-back")


def test_spurify_starts_on_the_street(spur_offline, monkeypatch, capsys):
    data, track, peaks = spur_network()
    lot = (43.9992, -72.0)
    east = (lot[0], -71.9995)  # where a path leaves the street for the loop, 40 m east of the lot
    street = [(lot[0], -72.0005), lot, east]  # a street 90 m south of the loop's start
    extra = osm([(9, street, {"highway": "residential", "name": "Lot Rd"}), (10, [east, (44.0, -72.0)], PATH)])
    ids = {e["id"] for e in data["elements"] if e["type"] == "node"}
    data["elements"] += [dict(e, id=e["id"] + 10**6) if e["type"] == "node" else
                         dict(e, nodes=[n + 10**6 for n in e["nodes"]]) for e in extra["elements"]]
    # the connector's north end must be the loop's own start node
    trail_start = next(e["id"] for e in data["elements"] if e["type"] == "node" and (e["lat"], e["lon"]) == (44.0, -72.0))
    for e in data["elements"]:
        if e["type"] == "way" and e["id"] == 10:
            e["nodes"][-1] = trail_start
    monkeypatch.setattr(core, "fetch_osm", lambda *a, **k: data)
    start = np.array([list(lot)])
    s = vm.spurify(np.vstack([start, track, start]), extra_mi=1, time_limit_s=10, workers=2)
    assert core._haversine(s.route.lat[0], s.route.lon[0], *lot) < 5  # starts at the lot, not on the path 40 m off


@pytest.mark.parametrize("body, expect", [
    ("<gpx><trk><trkseg><trkpt lat='42.0' lon='-74.0'/><trkpt lat='42.1' lon='-74.1'/></trkseg></trk></gpx>", 2),
    ('<gpx><trk><trkseg><trkpt lat="42" lon="-74"/><trkpt lon="-74.1" lat="42.1"/><trkpt lat="42.2" lon="-74.2"/>'
     '</trkseg></trk></gpx>', 3),
    ('<gpx><trk><trkseg><trkpt lat="4.2e1" lon="-7.4e1"/><trkpt lat="42.1" lon="-74.1"/></trkseg></trk></gpx>', 2),
    ('<gpx><rte><rtept lat="1" lon="1"/></rte><trk><trkseg><trkpt lat="42" lon="-74"/><trkpt lat="42.1" lon="-74"/>'
     '</trkseg></trk></gpx>', 2),
])
def test_read_gpx_formats(tmp_path, body, expect):
    p = tmp_path / "r.gpx"
    p.write_text(body)
    pts = vm.read_gpx(p)
    assert len(pts) == expect and pts[0][0] == 42.0


def test_read_gpx_bad_files(tmp_path):
    p = tmp_path / "x.gpx"
    p.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00")
    with pytest.raises(vm.VertmaxxerError):
        vm.read_gpx(p)
    with pytest.raises(vm.VertmaxxerError):
        vm.read_gpx(tmp_path / "missing.gpx")


def test_minimize_seeds_the_flattest_route_long_enough():
    e = lambda u, v, L, var: dict(u=u, v=v, length=L, var=var, road_len=0)
    edges = [e("S", "A", 1000, 0), e("A", "B", 1500, 90), e("B", "C", 1500, 90), e("C", "A", 1500, 90),
             e("A", "D", 1500, 10), e("D", "E", 1500, 10), e("E", "A", 1500, 10)]
    lolli = core.TOPOLOGIES["lollipop"]
    flat = api._seed(edges, "S", 7000, lolli, 1609, 0.25, min_length=6500, minimize=True)
    assert list(flat) == [2, 0, 0, 0, 1, 1, 1]
    hilly = api._seed(edges, "S", 7000, lolli, 1609, 0.25)
    assert list(hilly) == [2, 1, 1, 1, 0, 0, 0]
    assert api._seed(edges, "S", 7000, lolli, 1609, 0.25, min_length=6900, minimize=True) is None  # 6.5 km max


def test_cli_checks_outputs_before_solving(monkeypatch, tmp_path):
    def solve(*a, **k):
        raise AssertionError("solved")
    monkeypatch.setattr(cli, "find_route", solve)
    for extra in (["-o", str(tmp_path / "no" / "x.gpx")], ["--plot", str(tmp_path / "p.xyz")]):
        with pytest.raises(SystemExit) as e:
            cli.main(["--start", "44,-72", "--distance", "5"] + extra)
        assert e.value.code == 2


def test_unmarked_ways():
    herd = {"highway": "path", "name": "Herd Path", "informal": "yes"}
    data = osm([(1, PTS[:5], PATH), (2, PTS[4:9], herd)])
    names = {n for e in core.build_graph(data, True, None, [PTS[0]])[0] for n in e["names"]}
    assert "Herd Path (unmarked)" in names
    ways = {e["way"] for e in core.build_graph(data, True, None, [PTS[0]], marked_only=True)[0]}
    assert ways == {1}


@pytest.mark.parametrize("kw", [dict(start=(float("nan"), -72.0)), dict(start=(141.4, -73.9)),
                                dict(start=(-73.96568, 41.42698)), dict(distance_mi=float("inf")),
                                dict(seg_max_m=0), dict(time_limit_s=-5), dict(workers=-1),
                                dict(max_road_fraction=-0.5), dict(min_loop_frac=1.5),
                                dict(distance_mi=0.3, topology="loop"), dict(turn_penalty_ft=-1)])
def test_bad_inputs_are_option_errors(kw):
    kw = dict(dict(start=(44.0, -72.0), distance_mi=5), **kw)
    with pytest.raises(vm.OptionError):
        vm.find_route(**kw)


def test_start_as_text(offline):
    offline(osm([(1, PTS, PATH)]))
    assert run(f"{PTS[0][0]},{PTS[0][1]}", 2, "out-and-back").shape == "out-and-back"


def test_cli_road_fraction_with_roads():
    with pytest.raises(SystemExit) as e:
        cli.main(["--start", "44,-72", "--distance", "5", "--roads", "--max-road-fraction", "0"])
    assert e.value.code == 2


def test_spurify_negative_extra():
    with pytest.raises(vm.OptionError):
        vm.spurify(np.array([[44.0, -72.0], [44.01, -72.0]]), extra_mi=-2)


def test_gpx_names_are_escaped(tmp_path):
    p = tmp_path / "r.gpx"
    core.write_gpx(p, dict(lat=[44.0, 44.1], lon=[-72.0, -72.0], z_raw=[1.0, 2.0]), "Burroughs & Slide <loop>")
    import xml.etree.ElementTree as ET
    assert ET.parse(p).getroot().find(".//{*}name").text == "Burroughs & Slide <loop>"


def test_spurify_output_names(monkeypatch, tmp_path):
    written = []

    class Fake:
        route = type("R", (), dict(write_gpx=lambda self, path, name: written.append((path, name)), gain_ft=0,
                                   distance_mi=1, proven=False))()
        side_trips, base_gain_ft, base_distance_mi = [], 0, 1
    monkeypatch.setattr(cli, "spurify", lambda *a, **k: Fake())
    cli.spurify_main([str(tmp_path / "Route.GPX"), "--extra", "1"])
    cli.spurify_main([str(tmp_path / "a.gpx"), "--extra", "1", "-o", str(tmp_path / "out")])
    assert written == [(str(tmp_path / "Route_spurred.gpx"), "Route_spurred"), (str(tmp_path / "out"), "out")]


def test_short_legs_fold_into_the_one_before():
    legs = [["Trail A", 1000], ["service", 10], ["footway", 5], ["Trail A", 200], ["Trail B", 500]]
    assert cli._legs(legs) == [["Trail A", 1215], ["Trail B", 500]]


# ---------------------------------------------------------------- adversarial round 2

def test_street_start_snaps_to_the_street_network_not_a_cut_off_lane():
    lane = [(44.0, -72.0002), (44.0, -71.9998)]  # a one-block lane, joined to the rest only by a highway
    grid = [(44.0006, -72.003 + 0.0005 * i) for i in range(13)]  # a street 67 m north, part of a grid
    data = osm([(1, lane, {"highway": "service", "name": "Lane"}),
                (2, [(43.999, -72.0), lane[0]], {"highway": "primary", "name": "Blvd"}),
                (3, grid, {"highway": "residential", "name": "Park Ave"}),
                (4, [grid[0], (44.002, -72.003)], {"highway": "residential", "name": "West St"}),
                (5, [grid[-1], (44.002, -71.997)], {"highway": "residential", "name": "East St"})])
    _, ids = core.build_graph(data, True, None, [(44.0, -72.0)], snap_roads=(0,))
    names = {w["tags"]["name"] for w in data["elements"] if w["type"] == "way" and ids[0] in w["nodes"]}
    assert names == {"Park Ave"}


def test_cache_reuse_takes_the_smallest_covering_download_clipped(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    near, far = osm([(1, PTS[:3], PATH)]), osm([(2, [(44.3, -72.0), (44.31, -72.0)], PATH)])
    small = dict(elements=near["elements"])
    big = dict(elements=near["elements"] + [dict(e, id=e["id"] + 1000) if e["type"] == "node" else
                                            dict(e, nodes=[n + 1000 for n in e["nodes"]]) for e in far["elements"]])
    for name, data, r in (("osm_big", big, 50000.0), ("osm_small", small, 6000.0)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
        (tmp_path / f"{name}.meta.json").write_text(json.dumps(dict(points=[[44.0, -72.0]], radius=r, roads=True,
                                                                    way_ids=[])))
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("download")))
    got = core.fetch_osm([(44.0, -72.0)], 3000, True)
    assert {e["id"] for e in got["elements"] if e["type"] == "way"} == {1}
    (tmp_path / "osm_small.meta.json").unlink()
    got = core.fetch_osm([(44.0, -72.0)], 3000, True)  # from the big one, clipped to 3 km
    assert {e["id"] for e in got["elements"] if e["type"] == "way"} == {1}


def test_empty_peaks_answer_is_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(core.requests, "post", lambda *a, **k: Reply({"elements": []}))
    core._overpass("q", "peaks")
    assert list(tmp_path.glob("peaks_*.json"))


@pytest.mark.parametrize("reverse", [False, True])
def test_spurify_keeps_the_tracks_direction(spur_offline, reverse):
    track = spur_offline[::-1] if reverse else spur_offline
    s = vm.spurify(track, extra_mi=3, time_limit_s=10, workers=2)
    along = np.r_[0, np.cumsum(core._haversine(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1]))]
    q = track[np.searchsorted(along, along[-1] / 4)]  # the track's quarter-way point
    i = np.argmin(core._haversine(*q, s.route.lat, s.route.lon))
    assert s.route.dist[i] / s.route.dist[-1] < 0.5  # comes early in the output too, not late (reversed)


def test_two_loop_shapes_get_longer_by_default(offline, monkeypatch):
    offline(osm([(1, PTS[:21], PATH)]))
    seen = []

    def solve(edges, starts, ends, budget, topology, min_loop, time_limit, *a, **k):
        seen.append(time_limit)
        raise vm.VertmaxxerError("stop")
    monkeypatch.setattr(core, "solve", solve)
    for shape in ("figure-8", "loop"):
        with pytest.raises(vm.VertmaxxerError):
            vm.find_route(PTS[0], 5, shape, workers=2)
    assert seen == [600.0, 120.0]


def test_negative_coordinates_after_start():
    assert cli._attach_negative(["--start", "-73.9,41.4", "--distance", "5"]) == ["--start=-73.9,41.4", "--distance", "5"]
    with pytest.raises(SystemExit) as e:
        cli.main(["--start", "-73.96568,41.42698", "--distance", "5"])
    assert e.value.code == 2  # "check the order", not "expected one argument"


def test_api_niceties():
    import inspect
    assert "quiet" in inspect.signature(vm.find_route).parameters
    assert api._points(["42.03545,-74.35961", "41.42698,-73.96568"]) == [(42.03545, -74.35961), (41.42698, -73.96568)]
    with pytest.raises(vm.OptionError, match="outside the US"):
        api._points((51.18, -115.57), "3dep")  # Banff, with 3DEP asked for


def test_loop_from_a_stub_walks_out_to_the_loop(offline):
    """A start at the end of an approach trail: the loop starts where the approach meets it."""
    A, approach, net = tri_and_approach()
    offline(net)
    r = run(approach[20], 5, "loop")  # 1 km down the approach
    assert r.shape == "loop" and "access path" in r.details
    assert core._haversine(r.lat[0], r.lon[0], *approach[20]) < 5 and r.closed


# ---------------------------------------------------------------- metric

def test_metric_inputs_reach_the_api_in_miles(monkeypatch):
    got = {}

    def capture(start, distance_mi=None, topology=None, **k):
        got.update(k, distance_mi=distance_mi)
        raise vm.VertmaxxerError("stop")
    monkeypatch.setattr(cli, "find_route", capture)
    with pytest.raises(SystemExit):
        cli.main(["--start", "44,-72", "--distance", "10", "--metric"])
    assert got["distance_mi"] == pytest.approx(10 / 1.609344) and got["min_loop_mi"] == 1.0
    with pytest.raises(SystemExit):
        cli.main(["--start", "44,-72", "--time", "1:00", "--gap", "6:00", "--metric", "--min-loop", "2"])
    assert got["gap"] == pytest.approx(6 * 1.609344) and got["min_loop_mi"] == pytest.approx(2 / 1.609344)


def test_metric_route_output(capsys):
    r = type("R", (), dict(shape="loop", details="10.00 km", end=None, distance_mi=10 / 1.609344, gain_ft=1000 * core.M_TO_FT,
                           gain_raw_ft=1100 * core.M_TO_FT, proven=True, legs=[["Long Trail", 10000.0]],
                           closed=True))()
    cli._print_route(r, cli._Units(True))
    out = capsys.readouterr().out
    assert "Distance 10.00 km, gain 1,000 m (unsmoothed 1,100 m), 100 m/km" in out and "10.00 km  Long Trail" in out


def test_metric_spurify(monkeypatch, tmp_path, capsys):
    got = {}

    class Fake:
        route = type("R", (), dict(write_gpx=lambda self, path, name: None, gain_ft=2000 * core.M_TO_FT,
                                   distance_mi=12 / 1.609344, proven=False))()
        side_trips = [vm.SideTrip(leaves_at_mi=1 / 1.609344, length_mi=2 / 1.609344, gain_ft=1000 * core.M_TO_FT,
                                  summits=["Peak"])]
        base_gain_ft, base_distance_mi = 1000 * core.M_TO_FT, 10 / 1.609344

    def fake(gpx, extra, budget, **k):
        got.update(extra=extra)
        return Fake()
    monkeypatch.setattr(cli, "spurify", fake)
    cli.spurify_main([str(tmp_path / "r.gpx"), "--extra", "2", "--metric"])
    out = capsys.readouterr().out
    assert got["extra"] == pytest.approx(2 / 1.609344)
    assert "1.00 km       2.00 km   1,000 m      500  Peak" in out and "After: 2,000 m over 12.00 km (+1,000 m)" in out


def test_metric_setting_sticks_until_imperial(monkeypatch, capsys):
    got = []

    def capture(start, distance_mi=None, topology=None, **k):
        got.append(distance_mi)
        raise vm.VertmaxxerError("stop")
    monkeypatch.setattr(cli, "find_route", capture)
    for flags in (["--metric"], [], ["--imperial"], []):
        with pytest.raises(SystemExit):
            cli.main(["--start", "44,-72", "--distance", "10"] + flags)
    km = pytest.approx(10 / 1.609344)
    assert got == [km, km, 10, 10]
    assert "from now on" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        cli.main(["--start", "44,-72", "--distance", "10", "--metric", "--imperial"])
    assert e.value.code == 2


def test_start_at_a_street_corner_shared_with_a_path_stays_there(offline, capsys):
    """A crosswalk stub shares the corner node with the street; a bigger park path network lies 200 m off. The
    start stays at the corner."""
    corner = (44.0, -72.0)
    street = [(44.0, -72.002), corner, (44.0, -71.998)]
    stub = [corner, (43.9999, -72.0)]
    park = [(44.0018 + 0.0002 * i, -72.0) for i in range(20)]
    offline(osm([(1, street, {"highway": "residential", "name": "21st St"}), (2, stub, {"highway": "footway"}),
                 (3, park, {"highway": "footway", "name": "Park Path"}),
                 (4, [(44.0, -71.998), park[0]], {"highway": "residential", "name": "Ave"})]))
    try:  # only the snap matters here
        run(corner, 0.05, "out-and-back", roads=True)
    except vm.VertmaxxerError:
        pass
    assert "snapped 0 m" in capsys.readouterr().out



# ---------------------------------------------------------------- adversarial round 3

def town(main_primary=True):
    """Two blocks of streets either side of a main street, joined only across it."""
    tags = {"highway": "primary" if main_primary else "residential", "name": "Main St"}
    main = [(44.0, -72.006 + 0.001 * i) for i in range(13)]
    ways = [(1, main, tags)]
    for k, (side, i, j) in enumerate(((1, 2, 10), (-1, 3, 9))):  # the blocks meet Main St at different corners
        a, b = (44.0 + side * 0.004, main[i][1]), (44.0 + side * 0.004, main[j][1])
        ways += [(10 + k, [main[i], a], {"highway": "residential", "name": f"West {k}"}),
                 (20 + k, [a, b], {"highway": "residential", "name": f"Back {k}"}),
                 (30 + k, [b, main[j]], {"highway": "residential", "name": f"East {k}"})]
    return osm(ways)


def test_road_runs_along_a_primary_main_street(offline):
    offline(town())
    with pytest.raises(vm.VertmaxxerError, match="--primary-roads"):
        run((44.004, -72.0), 2.5, "loop", roads_only=True)
    assert run((44.004, -72.0), 2.5, "loop", roads_only=True, primary_roads=True).shape == "loop"


def test_walk_from_town_to_the_trails_is_free_of_the_road_cap(offline):
    """A 3 km trail loop at the top of a 1 km approach trail, 1.5 km of street from the start: the street is the
    way to the trails, so it doesn't count against the 10% road cap (it's half the road the route would allow)."""
    A = (44.0, -72.0)
    B = (A[0] + 1000 / M_PER_DEG, A[1])
    C = (A[0] + 500 / M_PER_DEG, A[1] + 1000 / (M_PER_DEG * np.cos(np.radians(44))))
    seg = lambda p, q: [(p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])) for f in np.linspace(0, 1, 21)]
    approach = line(*A, 1, south=True)
    street = line(*approach[-1], 1.5, south=True)
    offline(osm([(1, seg(A, B) + seg(B, C)[1:] + seg(C, A)[1:], {"highway": "path", "name": "Triangle"}),
                 (2, approach, {"highway": "path", "name": "Approach"}),
                 (3, street, {"highway": "residential", "name": "Village Rd"})]))
    r = run(street[-1], 5.5, "lollipop")  # 2 x (1.5 + 1) + 3.2 = 8.2 km; 3 km of it street
    assert r.shape == "lollipop" and any(n == "Village Rd" for n, _ in r.legs)


def test_access_path_counts_toward_road_cap_and_gain(offline, capsys):
    e = lambda u, v, L, road=0.0: dict(u=u, v=v, length=L, var=10.0, road_len=road, gap=(L, L), z=np.array([0.0, 0.0]))
    edges = [e("A", "B", 1000), e("B", "C", 1000), e("C", "A", 1000)]
    tri = core.TOPOLOGIES["loop"]
    ok = core.solve(edges, ["A"], None, 6000, tri, 0, 10, 1, False, max_road_frac=0.1, start_cost=[2000],
                    start_road=[0.0])
    assert ok is not None
    with pytest.raises(vm.VertmaxxerError):  # 2 km of road in a 5 km route: over a 10% cap
        core.solve(edges, ["A"], None, 6000, tri, 0, 10, 1, False, max_road_frac=0.1, start_cost=[2000],
                   start_road=[2000.0])
    capsys.readouterr()
    core.solve(edges, ["A"], None, 6000, tri, 0, 10, 1, False, start_cost=[2000], start_gain=[100.0])
    final = [ln for ln in capsys.readouterr().out.splitlines() if "optimal:" in ln][-1]
    assert "gain 377 ft" in final  # 15 m round the loop + 100 m out and back = 115 m


def test_no_network_says_so_quickly(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(core, "_DEAD_MIRRORS", set())
    calls = []
    monkeypatch.setattr(core.time, "sleep", lambda s: None)

    def down(*a, **k):
        calls.append(1)
        raise core.requests.ConnectionError("no route to host")
    monkeypatch.setattr(core.requests, "post", down)
    with pytest.raises(vm.VertmaxxerError, match="internet connection"):
        core._overpass("q", "osm")
    assert len(calls) == len(core.OVERPASS_URLS)


def test_us_territories_accepted():
    assert api._points([(18.34, -64.93), (13.44, 144.79)])  # USVI, Guam


def test_unnamed_legs_fold():
    legs = [["Long Trail", 2000], ["service", 60], ["gap in map data", 5], ["footway", 70], ["Ridge Trail", 50]]
    assert cli._legs(legs) == [["Long Trail", 2135], ["Ridge Trail", 50]]



def test_elevation_source_outside_the_us():
    assert api._dem_for([(42.0, -74.0)], "auto") == "3dep"
    assert api._dem_for([(46.56, 8.0)], "auto") == "terrarium"  # the Alps
    assert api._points((46.56, 8.0), "auto") == [(46.56, 8.0)]
    with pytest.raises(vm.OptionError, match="swap"):
        api._points((-73.96568, 41.42698), "auto")


def test_traffic_island_doesnt_count_as_a_trail():
    p = (44.0, -72.0)
    island = [(44.0001, -72.0001), (44.0001, -72.0)]  # 8 m of footway in the street, 11 m away
    street = [(44.0, -72.0003), (44.00012, -71.99995)]  # nearest node 13 m away
    park = [(44.002 + 0.0002 * i, -72.0) for i in range(20)]  # a real path network 220 m off
    data = osm([(1, island, {"highway": "footway", "footway": "traffic_island"}),
                (2, street, {"highway": "secondary", "name": "5th Avenue"}),
                (3, park, {"highway": "footway", "name": "Park Path"})])
    assert api._nearer_street(data, p)


def test_out_and_back_seed_is_the_climbiest_path_that_fits():
    e = lambda u, v, L, var, road=0.0: dict(u=u, v=v, length=L, var=var, road_len=road)
    edges = [e("S", "A", 500, 10), e("A", "B", 500, 80), e("S", "C", 500, 30), e("C", "D", 2000, 400)]
    oab = core.TOPOLOGIES["out-and-back"]
    assert list(api._seed(edges, "S", 2500, oab, 1609, 0.25)) == [2, 2, 0, 0]  # S-C-D doesn't fit 2.5 km
    assert list(api._seed(edges, "S", 6000, oab, 1609, 0.25)) == [0, 0, 2, 2]
    assert list(api._seed(edges, "S", 6000, core.TOPOLOGIES["any"], 1609, 0.25)) == [0, 0, 2, 2]


def test_walks_to_every_trail_network_in_reach_are_free(offline):
    """A small path network (1.7 km) is nearest to the start; the big one is farther. The walk to the big one is free
    too, so a route that needs it still fits a 10% road cap."""
    A = (44.0, -72.0)
    B = (A[0] + 1000 / M_PER_DEG, A[1])
    C = (A[0] + 500 / M_PER_DEG, A[1] + 1000 / (M_PER_DEG * np.cos(np.radians(44))))
    seg = lambda p, q: [(p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])) for f in np.linspace(0, 1, 21)]
    approach = line(*A, 1, south=True)
    street = line(*approach[-1], 1.5, south=True)
    start = street[-1]
    small = [(start[0] + 0.0001 * i, start[1] + 0.003) for i in range(0, 161, 10)]  # 1.8 km north-south, 240 m east
    offline(osm([(1, seg(A, B) + seg(B, C)[1:] + seg(C, A)[1:], {"highway": "path", "name": "Triangle"}),
                 (2, approach, {"highway": "path", "name": "Approach"}),
                 (3, street, {"highway": "residential", "name": "Village Rd"}),
                 (4, [start, (start[0], start[1] + 0.003)], {"highway": "residential", "name": "Park Rd"}),
                 (5, small, {"highway": "path", "name": "Park Path"})]))
    r = run(start, 5.5, "lollipop")
    assert any(n == "Triangle" for n, _ in r.legs)


def test_spurify_reads_an_out_and_back_track(spur_offline, capsys):
    """Out along the loop's west side to a point between junctions, and back: the base route is that, twice."""
    track = spur_offline
    out = track[:31]  # 3/4 of the way up the west side (the side trail leaves at the midpoint)
    oab = np.vstack([out, out[-2::-1]])
    s = vm.spurify(oab, extra_mi=0.5, time_limit_s=10, workers=2)
    data, _, _ = spur_network()
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in data["elements"] if e["type"] == "node"}
    trail = np.array([nodes[n] for n in next(w for w in data["elements"] if w["type"] == "way" and w["id"] == 1)["nodes"]])
    expected = 2 * api._along(trail[:31])[-1] / core.MI_TO_M  # out along the zigzag trail to the turnaround, and back
    assert abs(s.base_distance_mi - expected) < 0.01 * expected
    assert "more than twice" not in capsys.readouterr().err



def test_units_not_saved_by_a_run_with_bad_options(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--start", "44,-72", "--distance", "-1", "--metric"])
    assert e.value.code == 2 and cli._saved_units() is None
    assert "--distance must be" in capsys.readouterr().err  # the CLI's name, not the API's distance_mi


def test_turnaround_of_an_out_and_back_track():
    out = np.array(line(44.0, -72.0, 3))
    assert api._turnaround(np.vstack([out, out[-2::-1]])) == len(out) - 1
    loop = np.array([(44.0 + 0.01 * np.sin(t), -72.0 + 0.01 * np.cos(t)) for t in np.linspace(0, 2 * np.pi, 200)])
    assert api._turnaround(loop) is None


def test_driveways_are_off_limits(offline):
    """A steep dead-end driveway off a street: with roads allowed, routes still don't run up it."""
    st = [(44.0, -72.003 + 0.0005 * i) for i in range(13)]
    loop = st + [(44.003, -71.997), (44.003, -72.003), st[0]]
    drive = [st[6], (44.0025, st[6][1])]  # 280 m up the fake slope
    offline(osm([(1, loop, {"highway": "residential", "name": "Lane Gate Road"}),
                 (2, drive, {"highway": "service", "service": "driveway"})]))
    r = run(st[0], 3, "any", roads=True)
    assert not any("service" in n for n, _ in r.legs)


def test_figure8_seed_is_two_loops_meeting_at_the_start():
    e = lambda u, v, L, var: dict(u=u, v=v, length=L, var=var, road_len=0)
    edges = [e("S", "A", 1000, 50), e("A", "B", 1000, 50), e("B", "S", 1000, 50),   # loop 1
             e("S", "C", 1000, 20), e("C", "D", 1000, 20), e("D", "S", 1000, 20),   # loop 2
             e("A", "C", 500, 5)]
    seed = api._seed(edges, "S", 7000, core.TOPOLOGIES["figure-8"], 1609, 0.25)
    assert list(seed) == [1, 1, 1, 1, 1, 1, 0]


def test_figure8_seed_when_the_start_is_on_one_loop_only():
    """One trail leaves the start: the second loop meets the first further along."""
    e = lambda u, v, L, var: dict(u=u, v=v, length=L, var=var, road_len=0)
    edges = [e("S", "A", 800, 50), e("A", "B", 1000, 50), e("B", "S", 800, 50),   # loop 1 through the start
             e("A", "C", 1000, 20), e("C", "D", 1000, 20), e("D", "A", 1000, 20)]  # loop 2 through A
    assert list(api._seed(edges, "S", 7000, core.TOPOLOGIES["figure-8"], 1609, 0.25)) == [1, 1, 1, 1, 1, 1]


def test_crossing_kept_where_it_is_the_only_link():
    """A street ends across from a trailhead and a mapped crosswalk joins them (as at Cold Spring's Foundry Dock):
    the crosswalk stays. One that only joins two sidewalks across a road is still dropped."""
    road = {"highway": "residential", "name": "The Boulevard"}
    crossing = {"highway": "footway", "footway": "crossing"}
    end = PTS[8]
    across = (end[0] + 20 / M_PER_DEG, end[1])
    data = osm([(1, PTS[:9], road), (2, [end, across], crossing), (3, line(*across, 1), PATH)])
    edges, _ = core.build_graph(data, True, None, [PTS[0]], snap_roads=(0,))
    assert {1, 2, 3} <= {e["way"] for e in edges}

    sidewalk = {"highway": "footway", "footway": "sidewalk"}
    west, east = (PTS[4][0], PTS[4][1] - 1e-4), (PTS[4][0], PTS[4][1] + 1e-4)
    data = osm([(1, PTS[:9], road), (4, [(PTS[0][0], west[1]), west, (PTS[8][0], west[1])], sidewalk),
                (5, [(PTS[0][0], east[1]), east, (PTS[8][0], east[1])], sidewalk), (6, [west, PTS[4], east], crossing)])
    edges, _ = core.build_graph(data, True, None, [PTS[0]], snap_roads=(0,))
    assert 6 not in {e["way"] for e in edges}


def test_bundled_links_join_a_road_end_to_a_trail(monkeypatch, tmp_path):
    """A street ends 11 m short of a trail with no shared node in OSM (Kemble Avenue and the Foundry Preserve in
    Cold Spring): a links file joins them. Without it they stay apart, as a road end isn't bridged automatically."""
    road = {"highway": "residential", "name": "Kemble Avenue"}
    end = PTS[8]
    trail_start = (end[0] + 11 / M_PER_DEG, end[1])
    data = osm([(1, PTS[:9], road), (2, line(*trail_start, 1), PATH)])
    ids = {(n["lat"], n["lon"]): n["id"] for n in data["elements"] if n["type"] == "node"}
    a, b = ids[(round(end[0], 7), round(end[1], 7))], ids[(round(trail_start[0], 7), round(trail_start[1], 7))]

    def joined():
        edges, _ = core.build_graph(data, True, None, [PTS[0]], snap_roads=(0,))
        G = nx.Graph([(e["u"], e["v"]) for e in edges])
        return b in G and nx.has_path(G, a, b)

    monkeypatch.setattr(core, "LINKS_DIR", tmp_path, raising=False)
    assert not joined()
    (tmp_path / "kemble.json").write_text(json.dumps({"links": [[a, b]]}))
    assert joined()


def test_turn_penalty_reaches_the_solver(offline, monkeypatch):
    """30 ft per turnaround by default, or as given; shapes that can't turn back mid-trail don't use it."""
    offline(osm([(1, PTS, PATH)]))
    got = []
    real = core.solve
    monkeypatch.setattr(core, "solve", lambda *a, **k: got.append(k["turn_penalty"]) or real(*a, **k))
    run(PTS[0], 4, "out-and-back")
    run(PTS[0], 4, "out-and-back", turn_penalty_ft=100)
    run(PTS[0], 4, "out-and-back", turn_penalty_ft=0)
    assert got == [pytest.approx(30 / core.M_TO_FT), pytest.approx(100 / core.M_TO_FT), 0.0]


def test_turn_penalty_option_in_the_users_units(monkeypatch):
    got = {}

    def capture(start, distance_mi=None, topology=None, **k):
        got.update(k)
        raise vm.VertmaxxerError("stop")
    monkeypatch.setattr(cli, "find_route", capture)
    for flags, ft in ((["--turn-penalty", "50"], 50), (["--turn-penalty", "10", "--metric"], 10 * core.M_TO_FT), ([], 30)):
        got.clear()
        with pytest.raises(SystemExit):
            cli.main(["--start", "44,-72", "--distance", "5"] + flags)
        assert got.get("turn_penalty_ft", 30) == pytest.approx(ft)


def test_sicko_preset(monkeypatch):
    """--sicko: free turnarounds every 50 m on unsmoothed elevation; options given explicitly still win."""
    got = {}

    def capture(start, distance_mi=None, topology=None, **k):
        got.clear()
        got.update(k)
        raise vm.VertmaxxerError("stop")
    monkeypatch.setattr(cli, "find_route", capture)
    for flags, want in (([], (30, 50, 500)), (["--sicko"], (0, 0, 50)),
                        (["--sicko", "--seg-max", "25", "--smooth", "10"], (0, 10, 25))):
        with pytest.raises(SystemExit):
            cli.main(["--start", "44,-72", "--distance", "5"] + flags)
        assert (got["turn_penalty_ft"], got["smooth_m"], got["seg_max_m"]) == want
