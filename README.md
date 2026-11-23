# SwisTrack 4.3

[![Build](https://github.com/correlllab/swistrack/actions/workflows/build.yml/badge.svg)](https://github.com/correlllab/swistrack/actions/workflows/build.yml)

SwisTrack is a tool for tracking robots, humans, animals and objects using a camera or a recorded video as input. Tracking pipelines are assembled from components (input, background subtraction, thresholding, blob detection, calibration, tracking, output) in a wxWidgets GUI, and can be run headless with `SwisTrack-batch`.

## History

SwisTrack was developed at EPFL (Distributed Intelligent Systems and Algorithms Laboratory) from 2006 and hosted on [SourceForge](https://sourceforge.net/projects/swistrack/) until 2016. The full SVN history is preserved in this repository. In 2022, Thomas Lochmatter ported it to OpenCV 4 in [d28b/swistrack](https://github.com/d28b/swistrack), which this repository is forked from. This fork adds a CMake build and continuous integration for Linux, macOS and Windows.

## Documentation

- [Wiki](https://github.com/d28b/swistrack/wiki) (SwisTrack 4.2)
- [SwisTrack Wikibook](https://en.wikibooks.org/wiki/SwisTrack) (original documentation, partly outdated)

## Modern tracking

SwisTrack 4.3 adds a tracking-by-detection stack based on current best practice in multi-object tracking:

| Component | What it does |
|---|---|
| **Multi-Object Tracking** (`MultiObjectTracking`) | Kalman filter (constant velocity) per object, optimal (Hungarian) association with Mahalanobis gating, [ByteTrack](https://arxiv.org/abs/2110.06864)-style two-stage association that uses low-confidence detections to bridge partial occlusions, and [OC-SORT](https://arxiv.org/abs/2203.14360)'s direction-consistency cost, recovery from the last observation and filter re-update after occlusions. Tracks go through a tentative → confirmed → lost life cycle. |
| — *known number of objects* | For experiments with a fixed group size (e.g. 10 fish): set *Max. number of tracks* and enable *Known number of objects*. IDs 0…N−1 are kept for the whole video; lost animals are re-identified instead of getting new IDs. |
| **Blob Detection (split touching objects)** (`BlobDetectionSplitting`) | Learns the area and shape of a single object, and splits blobs of touching animals into individual detections with a shape-constrained Gaussian mixture (correctly separates animals lying side by side, where k-means fails). |
| **Adaptive Background Model** (`BackgroundModel`) | MOG2 / KNN background subtraction: no background image needed, adapts to lighting, removes shadows, optional freeze after N frames. |
| **Marker Detection** (`MarkerDetection`) | AprilTag (36h11, 25h9, 16h5, 36h10) and ArUco markers: ID, position and orientation per marker. Marker IDs become track IDs directly, so robots never swap identities. See [Examples/Markers](Examples/Markers). |

On the `Examples/leurre` video (10 fish, 900 frames), the classic pipeline with `NearestNeighborTracking` produces 49 track fragments. `leurre.modern.swistrack` (blob splitting + multi-object tracking with a known number of objects) tracks all 10 fish in 99.9% of the frames with exactly 10 identities.

A typical pipeline:

```
InputFileAVI → ConvertInputToGray → BackgroundSubtractionGray / BackgroundModel → ThresholdGray
  → BlobDetectionSplitting → MultiObjectTracking → OutputTracksToFiles
```

The tracking algorithms are plain C++ classes without GUI dependencies (`Code/core/MultiObjectTracker`, `BlobSplitter`, `MarkerDetector`, `LinearAssignment`) and have unit tests (`ctest --test-dir build`).

## Building

SwisTrack needs a C++17 compiler, [CMake](https://cmake.org) 3.16 or newer, [OpenCV](https://opencv.org) 4 or 5, and [wxWidgets](https://wxwidgets.org) 3.0 or newer.

### Ubuntu / Debian / Raspberry Pi OS

```sh
sudo apt install build-essential cmake libopencv-dev libwxgtk3.2-dev   # libwxgtk3.0-gtk3-dev on older releases
git clone https://github.com/correlllab/swistrack.git
cd swistrack
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/SwisTrack
```

### macOS

```sh
brew install cmake opencv wxwidgets
git clone https://github.com/correlllab/swistrack.git
cd swistrack
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
open build/SwisTrack.app
```

### Windows

Install Visual Studio 2022 (with "Desktop development with C++") and [vcpkg](https://vcpkg.io). The dependencies are declared in `vcpkg.json` and installed automatically during configuration:

```bat
git clone https://github.com/correlllab/swistrack.git
cd swistrack
cmake -S . -B build -A x64 -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
cmake --build build --config Release
build\Release\SwisTrack.exe
```

The first configuration builds OpenCV and wxWidgets from source, which takes a while.

### Installing

`cmake --install build --prefix <folder>` installs the executables together with the component descriptions (`Components/`) and `default.swistrack`. SwisTrack finds these files in the working directory, next to the executable, in `SwisTrack.app/Contents/Resources` (macOS), in `../share/swistrack` relative to the executable (Linux), or in the folder named by the `SWISTRACK_HOME` environment variable.

### Build options

| Option | Default | Description |
|---|---|---|
| `SWISTRACK_BUILD_GUI` | `ON` | Build the GUI application |
| `SWISTRACK_BUILD_BATCH` | `ON` | Build `SwisTrack-batch` |
| `SWISTRACK_WITH_SQLITE3` | `OFF` | Build the SQLite3 output component |

The legacy Makefiles (`make`, `make osxbundle`) and the Visual Studio solution in `Code/MSVisualStudio` are still included.

## Example

`Examples/leurre` contains a recorded video of fish with a ready-made tracking pipeline:

```sh
cd Examples/leurre
../../build/SwisTrack-batch leurre.swistrack
```

## License

SwisTrack is distributed under the GNU General Public License v3. See [COPYING](COPYING).
