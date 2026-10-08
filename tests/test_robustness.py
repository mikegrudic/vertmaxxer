"""find_route, spurify and the CLIs on hand-built OSM data with a fake elevation model (no network)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import vertmaxxer as vm
from vertmaxxer import api, cli, core

M_PER_DEG = 111195.0


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
        self.data, self.status_code = data, status

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
    assert capsys.readouterr().out.count("snapped") == 2  # the route's ends, not every peak


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
