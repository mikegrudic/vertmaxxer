# vertmaxxer

Give it a starting point, a distance and a route shape. It finds the route with the most climbing on the
OpenStreetMap trail network, and writes a GPX file.

```
vertmaxxer --start 42.03545,-74.35961 --distance 16 --topology loop
```

Elevation comes from USGS 3DEP in the US and from AWS terrain tiles elsewhere (worldwide, built from national data where it's fine-grained and ~30 m SRTM otherwise); `--dem` picks one. The solver uses OR-Tools CP-SAT; the model is
described at the top of `src/vertmaxxer/core.py`.

## Install

```
pip install "vertmaxxer[plot] @ git+https://github.com/mikegrudic/vertmaxxer"
```

or, from a clone, `pip install -e ".[plot,test]"`. Python 3.10 or later; `[plot]` adds matplotlib for `--plot`.

Trail and elevation downloads are cached in `~/.cache/vertmaxxer` (set `VERTMAXXER_CACHE` to move it). A later
run from the same start at the same or a shorter distance reuses the trail download; a longer one, or a start
outside the area already downloaded, fetches again.

## Vertmaxx a route

1. **Get the start as `lat,lon`.** In Google Maps, right-click the spot and click the coordinates to copy them.
   Maps copies them with a space after the comma: delete it, or quote the pair (`--start "42.03545, -74.35961"`).
   The route starts on the nearest trail or street to that point. From a house, it walks the streets to the
   trails (the shortest way to each trail network in reach); those walks count toward the distance but not toward
   `--max-road-fraction`. To start on a particular
   trail, put the point on it.
2. **Pick a distance** in miles. This is a maximum; the route can come in shorter if the extra distance would add
   no climbing.
3. **Pick a shape:**

   | `--topology`   | Route |
   |----------------|-------|
   | `loop`         | One loop, nothing repeated |
   | `out-and-back` | Out and back along the same trail |
   | `lollipop`     | Out on a stem, around a loop, back down the stem (the default) |
   | `figure-8`     | Two loops through one crossing point, nothing repeated |
   | `dumbbell`     | A loop, a repeated connector, a second loop, then back |
   | `traverse`     | Point to point; needs `--end LAT,LON` or `--end-trailheads` |
   | `double-lollipop` | A stem to two separate loops |
   | `loop-spurs`   | One loop plus out-and-back side trips |
   | `spurred`      | Any closed route with out-and-back side trips |
   | `any`          | Whatever climbs most, as long as no trail is run more than twice |

   A route that fits none of these names (from `any`, say) is reported as shape `other`, with its loops and
   turnarounds described.

4. **Run it:**

   ```
   vertmaxxer --start 44.21787,-71.41128 --distance 20 --topology lollipop \
       -o crawford.gpx --plot crawford.png
   ```

   It prints the shape, the distance, the gain and a turn-by-turn list of trails, then writes the GPX and, with
   `--plot`, a map and elevation profile. A lollipop in a big trail network keeps improving for minutes: if the
   printed bound is far above the gain, run again with `--time-limit 600`.

   Another: the Burroughs Range loop from Woodland Valley, over Wittenberg, Cornell and Slide, with a walk on
   Oliverea Road back to the Phoenicia-East Branch Trail:

   ```
   vertmaxxer --start 42.03545,-74.35961 --distance 16 --topology loop
   ```

### The rules it follows

By default it plays by the same rules as the published survey:

- **Trails**, with up to 10% of the distance on roads (`--max-road-fraction`), for walks between trailheads.
  Road crossings of up to 50 m and the roads within 400 m of the start (its parking lots and access roads) don't
  count toward that. Highways (OSM primary and trunk) can be crossed but not followed.
- **Closed segments** bundled with the package (`src/vertmaxxer/data/closures/`) are avoided: closed trails and roads (such as NY 9D at Breakneck), and abandoned
  trails still in OSM.
- **Traverses** with `--end-trailheads` don't end on the Mount Washington Auto Road, at the Mount Washington summit,
  or on Breakneck Road (NY 9D).
- **Turnarounds** (in out-and-backs and `any` routes) each count as 30 ft of climbing given up, so a route
  doesn't nip a few meters up a side street for a few feet. Dead-end road stubs under 100 m (alleys, lot
  entrances) and duplicate short ways are ignored.
- **Loops** are at least 1 mi and a quarter of the route long. A traverse to `--end-trailheads` ends at least 1 mi
  from its start (`--min-end-dist`); an `--end` at the start is refused, since that's a loop.
- **Starts and ends** snap to the nearest trail or street, with a warning past 200 m and an error past 1 km.

### Useful options

| Option | What it does |
|--------|--------------|
| `--time H:MM --gap M:SS` | A time budget at a grade-adjusted pace per mile, instead of `--distance` (see below) |
| `--time H:MM --pace M:SS` | A time budget at a plain pace per mile: the same as `--distance` time ÷ pace |
| `--metric` / `--imperial` | Kilometers and meters (distances, `--pace` and `--gap` per km, and all output), or miles and feet. The choice is saved (in `~/.config/vertmaxxer/settings.json`) and applies to later runs of both commands until changed |
| `-o FILE.gpx` | Where to write the route (default `vertmaxxer.gpx`) |
| `--plot FILE.png` | Also draw a map and elevation profile |
| `--roads` | Allow roads as well as trails, any amount |
| `--roads-only` | Roads only, paved or dirt: no trails, tracks, driveways or parking aisles. Highways (OSM primary and trunk) can be crossed but not followed |
| `--paved-only` | Leave out roads tagged as unpaved (gravel, dirt, ...). Many roads have no surface tag; those count as paved |
| `--ways FILE` | Add or exclude particular OSM ways (see Road runs below) |
| `--minimize` | Find the flattest route instead, covering at least 98% of `--distance`. Works well for loops and road runs. Elsewhere it starts from the hilliest route and works down, so a lollipop in a big trail network can stay far from the flattest; give it a longer `--time-limit` |
| `--primary-roads` | Allow running along primary roads (often a town's main street), not just across them |
| `--max-road-fraction F` | At most this share of the distance on roads (default 0.1; 0 for trails only) |
| `--trailhead-roads M` | Walkable roads around the start, in meters (default 400; 0 for none) |
| `--turn-penalty GAIN` | Climb a turnaround must be worth, in feet (meters with `--metric`; default 30, 0 for free). Applies to shapes that can turn back mid-trail (`out-and-back`, `any`, ...); higher keeps routes from nipping up short side trips |
| `--any-end` | Allow traverses to end on the Mount Washington Auto Road, at its summit, or on Breakneck Road |
| `--time-limit S` | Seconds to search (default 120; 600 for `figure-8`, `dumbbell` and `double-lollipop`). Long routes and `any` often improve with 600 or more |
| `--end LAT,LON` | Finish here (with `--topology traverse`) |
| `--end-trailheads` | Finish at whichever trailhead gives the most gain (with `--topology traverse`) |
| `--closures FILE` | Also avoid the closed segments listed in FILE (format as in `src/vertmaxxer/data/closures/`) |
| `--max-sac N` | Skip trails rated harder than SAC grade TN (1-6) |
| `--marked-only` | Skip herd paths and other informal or unmarked ways, which the turn list otherwise labels "(unmarked)" |
| `--start` again | Give several starts; the solver uses whichever is best |

`vertmaxxer --help` lists the rest.

### Time instead of distance

`--time 5:00 --gap 13:00` finds the hilliest route you can run in five hours at a 13:00/mi grade-adjusted pace,
using Strava's grade-adjusted pace (GAP) model: each stretch costs its flat-equivalent distance, so climbs
and steep descents use up more of the budget than flat trail does.

```
vertmaxxer --start 42.03545,-74.35961 --time 5:00 --gap 13:00 --topology loop
```

It prints the route's grade-adjusted distance and time, and for a loop the time run in reverse too. The solver
doesn't choose a direction for a stretch run once, so it charges the average of the two directions; a loop's
time in the direction you run it can differ from that by a few percent. GAP is a running model: on very steep
or technical trail, expect to be slower than it says.

### Reading the result

- **Gain** is computed from elevation smoothed over about 50 m, which removes noise from the elevation data.
  It reads about 9% lower than CalTopo. The figure in parentheses is the gain without smoothing.
- **The search usually runs until the time limit** rather than proving its route is the best possible. A longer
  `--time-limit` sometimes finds more.
- **If it prints "requested X but the route is a Y"**, the best route it found has a simpler shape, for example
  a lollipop whose loop shrank away. Try another shape, or a different distance.

## Road runs

`--roads-only` plans a run on streets. OSM sorts ways by type, not surface: `highway=track` (most forest and farm
roads) counts as a trail and is left out, while a gravel town road counts as a road and is kept unless you add
`--paved-only`.

A `--ways` file fine-tunes the network: `include` adds ways of any type, even ones normally left out such as
sidewalks (a cemetery lane, a pedestrian tunnel), and `exclude` drops ways (a private drive). Way ids come from openstreetmap.org: click a way and read the id from the URL.
Dead ends of included paths are joined to the nearest street within 30 m, since the graph leaves out sidewalks.
`examples/cold_spring_road_runs.json` is the Cold Spring setup: the cemetery lanes and the Main Street tunnel, without
the private drives.

```
# the hilliest 10K loop from the Cold Spring bandstand, then the flattest
vertmaxxer --start 41.41602,-73.96122 --roads-only --ways examples/cold_spring_road_runs.json \
    --distance 6.214 --topology loop -o cs_hilly.gpx
vertmaxxer --start 41.41602,-73.96122 --roads-only --ways examples/cold_spring_road_runs.json \
    --distance 6.214 --topology loop --minimize -o cs_flat.gpx
```

## Add summit side trips to a route

To add out-and-back side trips to summits to a route you already have (from this tool, CalTopo or a watch),
use `vertmaxxer-spurify`. It keeps your route and adds out-and-back trips that turn around only at named peaks
(or where your route already turns around):

```
vertmaxxer-spurify my_route.gpx --extra 3        # up to 3 more miles
vertmaxxer-spurify my_route.gpx --budget 30      # or a total distance
```

It writes `my_route_spurred.gpx` (or `-o FILE`) and prints a table of the side trips added, in route order, with
their length, gain and summits. `--extra` counts from the length of trail your track follows, which is often a
little longer than the track itself. If it warns that the matched length is off, which can happen with a noisy
watch track, raise `--match-m`.

A trip listed as "(connector, no summit)" is trail run out and back without a summit of its own: usually the
shared approach to several side trips, or a short climb worth its miles. Spurify's "Before" gain can differ by a
percent or so from `vertmaxxer`'s figure for the same route, because elevation smoothing is pinned at trail
junctions and the two build their networks with different junctions.

## Python API

The command-line tools are thin wrappers over two functions. Their numbers are in miles and feet (`Route` also has
`distance_km` and `gain_m`):

```python
import vertmaxxer as vm

r = vm.find_route((42.03545, -74.35961), 16, "loop")  # same options as the CLI, as keyword arguments
print(r.shape, r.distance_mi, r.gain_ft, r.proven)
for name, meters in r.legs:
    print(name, meters)
r.write_gpx("burroughs.gpx")
r.plot("burroughs.png")  # needs matplotlib

s = vm.spurify("my_route.gpx", extra_mi=3)
print(s.base_gain_ft, s.route.gain_ft, s.side_trips)
```

`find_route` and `spurify` raise `vertmaxxer.VertmaxxerError` when no route fits or a data source fails, and
`vertmaxxer.OptionError` (a `ValueError`) for invalid options. Both print progress; pass `quiet=True` to silence it.

## When it fails

| Message | What to do |
|---------|------------|
| `All Overpass servers failed` | The OpenStreetMap servers are busy. Try again in a few minutes |
| HTTP 504 from the USGS elevation service | Add `--dem terrarium` to use AWS terrain tiles instead |
| `No route found within the 120 s time limit` | The search ran out of time before finding any route; give it a longer `--time-limit`. Big networks and the two-loop shapes need minutes |
| `No feasible route found` | No route of that shape fits the distance. A loop may need a longer road walk between trailheads: raise `--max-road-fraction`. Otherwise try more miles or another shape |
| A start snapped hundreds of meters away | The point isn't near a mapped trail. Move it onto the trail |
