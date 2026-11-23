# SwisTrack 4.2

[![Build](https://github.com/correlllab/swistrack/actions/workflows/build.yml/badge.svg)](https://github.com/correlllab/swistrack/actions/workflows/build.yml)

SwisTrack is a tool for tracking robots, humans, animals and objects using a camera or a recorded video as input. Tracking pipelines are assembled from components (input, background subtraction, thresholding, blob detection, calibration, tracking, output) in a wxWidgets GUI, and can be run headless with `SwisTrack-batch`.

## History

SwisTrack was developed at EPFL (Distributed Intelligent Systems and Algorithms Laboratory) from 2006 and hosted on [SourceForge](https://sourceforge.net/projects/swistrack/) until 2016. The full SVN history is preserved in this repository. In 2022, Thomas Lochmatter ported it to OpenCV 4 in [d28b/swistrack](https://github.com/d28b/swistrack), which this repository is forked from. This fork adds a CMake build and continuous integration for Linux, macOS and Windows.

## Documentation

- [Wiki](https://github.com/d28b/swistrack/wiki) (SwisTrack 4.2)
- [SwisTrack Wikibook](https://en.wikibooks.org/wiki/SwisTrack) (original documentation, partly outdated)

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
