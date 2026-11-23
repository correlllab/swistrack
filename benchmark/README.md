# Tracking benchmark

Measures tracking quality objectively: synthetic videos with exact ground truth are tracked by
`SwisTrack-batch`, and the output is scored with standard multi-object tracking metrics.

```
swistrack-synth  ->  video.avi, background.png, groundtruth.csv, nn/mot/split/known.swistrack
SwisTrack-batch  ->  testing/<tracker>.csv         [OutputTracksCSV component]
swistrack-eval   ->  MOTA, IDF1, HOTA, ...
```

## Running

Build SwisTrack first (the script builds nothing), then pass the build folder:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j10
benchmark/run_benchmark.sh build                          # all scenarios, seed 1, 600 frames
benchmark/run_benchmark.sh build --seeds "1 2 3"          # mean over 3 seeds (recommended)
benchmark/run_benchmark.sh build --scenarios "fish fast" --frames 300 --keep /tmp/bench
```

Options: `--frames N`, `--seeds "S1 S2 ..."` (or `--seed S`), `--threshold PX` (match distance, default 10),
`--scenarios "..."`, `--trackers "nn mot split known"` (nn: NearestNeighborTracking; mot: MultiObjectTracking; split: + BlobDetectionSplitting; known: + known number of objects), `--codec mjpg|ffv1`, `--keep DIR` (keep videos, logs, CSVs and
per-run `*.metrics.json`; otherwise a temporary folder is used and deleted).

The script prints a markdown table (scenario x tracker: MOTA, IDF1, HOTA, IDSW, FP, FN; means over the seeds)
on stdout and progress on stderr. `MultiObjectTracking` is evaluated only if
`Components/MultiObjectTracking.xml` exists next to `SwisTrack-batch` (or in `SwisTrackEnvironment/Components`);
`--trackers` overrides this. The exit status is non-zero if a run failed.

**Windows**: run the script from Git Bash (`bash benchmark/run_benchmark.sh build`); it finds `*.exe` files and
the `Release/` subfolder of multi-config (Visual Studio) builds.

The tools can also be used on their own:

```sh
build/swistrack-synth --scenario occlusion --n 8 --frames 600 --seed 3 --out /tmp/occ
cd /tmp/occ && /path/to/build/SwisTrack-batch nn.swistrack      # writes /tmp/occ/testing/nn.csv
build/swistrack-eval /tmp/occ/groundtruth.csv /tmp/occ/testing/nn.csv [--threshold 10] [--json]
```

`swistrack-eval` accepts any CSV with a header containing `frame,id,x,y` (extra columns are ignored), so it
can also score other trackers or hand-annotated ground truth.

## Scenarios

All scenarios render identical dark ellipses on a light, slightly textured background (anti-aliased by 4x4
supersampling, Gaussian sensor noise), moving as correlated random walks (smoothly varying turning rate and
speed) with reflecting walls. The generator is deterministic: the same seed gives the same files (it uses
`std::mt19937` with its own distributions; tiny floating-point differences between math libraries can still
make other platforms diverge).

| Scenario | Objects | What it tests |
|---|---|---|
| `fish` | 10, 18x7 px, ~2.5 px/frame, circular arena | baseline; close passes and crossings |
| `crowded` | 30, 11x5 px, smaller rectangular arena | high density, frequent merges |
| `occlusion` | 8, like fish, plus 6 static occluding bars | objects vanish for a few frames |
| `fast` | 8, ~10 px/frame, sudden 60-150 degree turns | motion prediction and gating |
| `noise` | 10, like fish, strong noise | false-positive specks (1 frame), flickering objects (missed detections) |

Ground truth (`groundtruth.csv`, `frame,id,x,y`):
- frames are numbered from 0, like `InputFileAVI`;
- coordinates use the convention of SwisTrack's blob detector: pixel (i, j) covers [i, i+1) x [j, j+1),
  so the top-left pixel center is (0.5, 0.5). With the generated configuration, the mean blob-to-truth distance is
  about 0.2 px (MJPG) or 0.1 px (`--codec ffv1`, lossless);
- occluded objects are listed only if at least half of their area is visible;
- flickering objects (noise scenario) are physically present, only rendered with very low contrast: they *are*
  listed, so missed detections count as FN;
- when objects touch, the blob detector reports a single blob. This is kept on purpose (it happens with real
  animals): both objects stay in the ground truth, so a merge costs at least one FN per frame.

The generated configurations use `InputFileAVI -> ConvertInputToGray -> BackgroundSubtractionGray
(SubBackgroundImage) -> ThresholdGray (half the object contrast) -> BlobDetectionMinMax (area 0.35x - 4x the
object area) -> tracker -> OutputTracksCSV`. `nn.swistrack` sets NearestNeighborTracking to
MaxDistance = max(25, 1.5 x max speed), MaxNumberOfTracks = 2N; `mot.swistrack` uses MultiObjectTracking with
its default parameters. `scenario.txt` lists the parameters and a few statistics.

`OutputTracksCSV` writes `frame,id,x,y,vx,vy`, one row per track updated in the current frame. A relative file
name is resolved against the *run folder*, like every other SwisTrack output: `testing/` next to the
configuration file for SwisTrack-batch (and test runs in the GUI), a timestamped folder in production mode.
Absolute paths are used as given.

## Metrics

A ground truth point and a hypothesis point match if their distance d is at most the threshold (default 10 px,
about half an object length). Definitions follow the reference implementations; see
`Code/tools/TrackingMetrics.h`.

- **CLEAR MOT** (Bernardin & Stiefelhagen 2008, as in py-motmetrics). Per frame, each object keeps its previous
  hypothesis if that hypothesis is present and still within the threshold; the remaining objects are matched
  optimally (maximum number of matches, then minimum total distance).
  - **MOTA** = 1 - (FN + FP + IDSW) / #ground-truth points. Dominated by detection errors.
  - **MOTP**: mean distance of the matches, in pixels (lower is better).
  - **IDSW**: matches whose hypothesis differs from the last one matched to that object.
  - **Frag**: interruptions of the tracking of an object (tracked, then untracked, then tracked again).
  - **MT / PT / ML**: objects tracked in >= 80%, 20-80%, < 20% of the frames in which they are present.
- **IDF1 / IDP / IDR** (Ristani et al. 2016): each ground truth trajectory is assigned to at most one hypothesis
  trajectory (globally optimal); IDTP counts the frames in which assigned pairs match. IDF1 = 2 IDTP / (#GT +
  #hypothesis points). Measures how long identities are kept: swapping two identities halfway gives IDF1 = 0.5.
- **HOTA** (Luiten et al. 2020, as in TrackEval) with the similarity max(0, 1 - d / threshold) in place of IoU.
  For each alpha in {0.05, ..., 0.95}, matches need similarity >= alpha;
  DetA = TP / (TP + FN + FP), AssA = mean association Jaccard index over the matches, HOTA = sqrt(DetA x AssA);
  the reported values are averages over alpha. LocA is the mean similarity of the matches (TrackEval convention:
  1 for an alpha without matches). HOTA balances detection and association.

For comparing trackers, look at IDF1, HOTA/AssA and IDSW: on these scenarios detection is almost perfect, so MOTA
is close to 1 for any tracker and mostly reflects merged blobs and noise. Use several seeds: identity metrics vary
a lot from one random scenario to another.

## Tests

`ctest --test-dir build` runs `TrackingMetrics` (hand-computed metric values for tiny cases: perfect tracking,
one FP, one FN, one ID switch, swapped identities, kept correspondences, optimal vs greedy matching, localization,
MT/ML/Frag, merged blobs) and `BenchmarkEndToEnd` (`benchmark/EndToEndTest.cmake`: a 150-frame `fish` scenario
through swistrack-synth, SwisTrack-batch with NearestNeighborTracking and swistrack-eval; checks MOTA > 0.5,
IDF1 > 0.3 and MOTP < 1 px, which also catches frame-number or coordinate-convention mismatches).

## Note on partially visible objects

The ground truth lists an object only while at least half of it is visible. `BlobDetectionSplitting` measures
blob areas by pixel count (slightly larger than the contour area used by `BlobDetectionMinMax`), so it also detects
objects that are 35-50% visible at the edge of an occluder. These count as false positives in the `occlusion`
scenario (about 40 per run), although they are real objects.
