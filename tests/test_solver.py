"""The solver on a small synthetic network (no downloads): two triangles of trail sharing the start."""
import numpy as np
import pytest

from vertmaxxer import core

# Node: (lat, lon, elevation m). Triangle A-B-C climbs 100 m, A-D-E 50 m; each about 3 km around.
NODES = {"A": (44.0, -72.0, 0.0), "B": (44.009, -72.0, 100.0), "C": (44.0045, -71.989, 0.0),
         "D": (43.991, -72.0, 50.0), "E": (43.9955, -71.989, 0.0)}


def network(road=()):
    edges = []
    for u, v in [("A", "B"), ("B", "C"), ("C", "A"), ("A", "D"), ("D", "E"), ("E", "A")]:
        (la1, lo1, z1), (la2, lo2, z2) = NODES[u], NODES[v]
        lat, lon = np.linspace(la1, la2, 11), np.linspace(lo1, lo2, 11)
        z = np.linspace(z1, z2, 11)
        s = np.r_[0, np.cumsum(core._haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]))]
        edges.append(dict(u=u, v=v, lat=lat, lon=lon, z=z, z_raw=z, s=s, length=float(s[-1]),
                          names=np.full(11, f"{u}{v}", dtype=object), roadpt=np.full(11, (u, v) in road),
                          road=(u, v) in road))
    return core.subdivide(edges, None)


def loop(edges, budget, **kw):
    m, start, end, _ = core.solve(edges, ["A"], None, budget, core.TOPOLOGIES["loop"], 0.0, 10, 1, False, **kw)
    route = core.assemble(edges, m, start, end)
    return {edges[k]["names"][0] for k in np.flatnonzero(m)}, np.sum(np.maximum(0, np.diff(route["z"])))


def test_max_gain_picks_the_higher_triangle():
    used, gain = loop(network(), 3500)
    assert used == {"AB", "BC", "CA"}
    assert gain == pytest.approx(100)


def test_minimize_picks_the_lower_triangle():
    edges = network()
    length = sum(e["length"] for e in edges[3:])
    used, gain = loop(edges, length, minimize=True, min_length=0.98 * length)
    assert used == {"AD", "DE", "EA"}
    assert gain == pytest.approx(50)


def test_road_fraction_keeps_off_roads():
    edges = network(road={("A", "B")})
    assert loop(edges, 3500)[0] == {"AB", "BC", "CA"}
    assert loop(edges, 3500, max_road_frac=0.0)[0] == {"AD", "DE", "EA"}


def test_budget_too_small_raises():
    with pytest.raises(core.VertmaxxerError):
        loop(network(), 1000)
