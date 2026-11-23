#!/usr/bin/env bash
# SwisTrack tracking benchmark: generates synthetic scenarios, runs SwisTrack-batch with each tracker,
# scores the results against the ground truth and prints a markdown table.
#
#   benchmark/run_benchmark.sh BUILD_DIR [--frames N] [--seeds "1 2 3"] [--threshold PX]
#                              [--scenarios "fish crowded occlusion fast noise"] [--trackers "nn mot split known"]
#                              [--codec mjpg|ffv1] [--keep DIR]
#
# BUILD_DIR must contain the (already built) SwisTrack-batch, swistrack-synth and swistrack-eval executables,
# and the Components folder next to SwisTrack-batch (the CMake build copies it there).
# See benchmark/README.md.

set -u
export LC_ALL=C

usage() {
	sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' >&2
	exit 2
}

[ $# -ge 1 ] || usage
BUILD_DIR=$1
shift

FRAMES=600
SEEDS=1
THRESHOLD=10
SCENARIOS="fish crowded occlusion fast noise"
CODEC=mjpg
KEEP_DIR=""
TRACKERS=""
while [ $# -gt 0 ]; do
	case "$1" in
		--frames) FRAMES=$2; shift 2 ;;
		--seed) SEEDS=$2; shift 2 ;;
		--seeds) SEEDS=$2; shift 2 ;;
		--threshold) THRESHOLD=$2; shift 2 ;;
		--scenarios) SCENARIOS=$2; shift 2 ;;
		--trackers) TRACKERS=$2; shift 2 ;;
		--codec) CODEC=$2; shift 2 ;;
		--keep) KEEP_DIR=$2; shift 2 ;;
		-h|--help) usage ;;
		*) echo "Unknown option: $1" >&2; usage ;;
	esac
done

# Locate an executable in the build folder (single- and multi-config generators, .exe on Windows)
find_exe() {
	for dir in "$BUILD_DIR" "$BUILD_DIR/Release" "$BUILD_DIR/RelWithDebInfo" "$BUILD_DIR/Debug"; do
		for name in "$1" "$1.exe"; do
			if [ -f "$dir/$name" ] && [ -x "$dir/$name" ]; then
				(cd "$dir" && printf '%s/%s\n' "$(pwd)" "$name")
				return 0
			fi
		done
	done
	echo "error: $1 not found in $BUILD_DIR (build it first)" >&2
	return 1
}

BATCH=$(find_exe SwisTrack-batch) || exit 1
SYNTH=$(find_exe swistrack-synth) || exit 1
EVAL=$(find_exe swistrack-eval) || exit 1

# By default, the MultiObjectTracking tracker is evaluated only if this SwisTrack build has it
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BATCH_DIR=$(dirname "$BATCH")
HAVE_MOT=0
if [ -f "$BATCH_DIR/Components/MultiObjectTracking.xml" ] || [ -f "$SCRIPT_DIR/../SwisTrackEnvironment/Components/MultiObjectTracking.xml" ]; then
	HAVE_MOT=1
fi
if [ -z "$TRACKERS" ]; then
	TRACKERS="nn"
	[ $HAVE_MOT -eq 1 ] && TRACKERS="nn mot split known"
fi

for tracker in $TRACKERS; do
	case $tracker in
		nn|mot|split|known) ;;
		*) echo "error: unknown tracker '$tracker' (nn, mot, split or known)" >&2; exit 2 ;;
	esac
done

if [ -n "$KEEP_DIR" ]; then
	mkdir -p "$KEEP_DIR" || exit 1
	WORK=$(cd "$KEEP_DIR" && pwd)
else
	WORK=$(mktemp -d 2>/dev/null || mktemp -d -t swistrack-benchmark)
	trap 'rm -rf "$WORK"' EXIT
fi

echo "Benchmark: frames=$FRAMES seeds='$SEEDS' threshold=${THRESHOLD}px codec=$CODEC trackers: $TRACKERS" >&2
case " $TRACKERS " in
	*" mot "*) ;;
	*) echo "MultiObjectTracking component not found: evaluating NearestNeighborTracking only." >&2 ;;
esac
echo "Working folder: $WORK" >&2

# Mean of a metric over the JSON lines of a results file
mean_of() {
	sed -n "s/.*\"$1\":\([^,}]*\).*/\1/p" "$2" | awk -v d="$3" '
		$1 == "null" { nulls++; next }
		{ sum += $1; n++ }
		END { if (n == 0 || nulls > 0) print "n/a"; else printf("%.*f\n", d, sum / n) }'
}

STATUS=0
for scenario in $SCENARIOS; do
	for seed in $SEEDS; do
		dir="$WORK/$scenario-seed$seed"
		if ! "$SYNTH" --scenario "$scenario" --frames "$FRAMES" --seed "$seed" --codec "$CODEC" --out "$dir" >&2; then
			echo "error: swistrack-synth failed for scenario $scenario" >&2
			STATUS=1
			continue
		fi

		for tracker in $TRACKERS; do
			results="$WORK/results-$scenario-$tracker.jsonl"
			# SwisTrack-batch resolves relative paths against the configuration folder and writes the
			# OutputTracksCSV file into the run folder "testing/"
			rm -f "$dir/testing/$tracker.csv"
			(cd "$dir" && "$BATCH" "$tracker.swistrack" > "$tracker.log" 2>&1)
			result="$dir/testing/$tracker.csv"
			if grep -q "ignored because there is no component" "$dir/$tracker.log"; then
				echo "$scenario/$tracker: component not available in this build (see $dir/$tracker.log)" >&2
				echo "unavailable" >> "$results"
				continue
			fi
			if [ ! -s "$result" ]; then
				echo "error: $scenario/$tracker (seed $seed) produced no output; SwisTrack-batch log:" >&2
				sed 's/^/    /' "$dir/$tracker.log" | head -20 >&2
				echo "failed" >> "$results"
				STATUS=1
				continue
			fi

			json=$("$EVAL" "$dir/groundtruth.csv" "$result" --threshold "$THRESHOLD" --json)
			if [ $? -ne 0 ] || [ -z "$json" ]; then
				echo "error: swistrack-eval failed for $scenario/$tracker (seed $seed)" >&2
				echo "failed" >> "$results"
				STATUS=1
				continue
			fi
			printf '%s\n' "$json" > "$dir/$tracker.metrics.json"
			printf '%s\n' "$json" >> "$results"
		done
	done
done

# Markdown table (means over the seeds)
COUNT_DECIMALS=0
[ "$(echo $SEEDS | wc -w)" -gt 1 ] && COUNT_DECIMALS=1
echo "| Scenario | Tracker | MOTA | IDF1 | HOTA | IDSW | FP | FN |"
echo "|---|---|---:|---:|---:|---:|---:|---:|"
for scenario in $SCENARIOS; do
	for tracker in $TRACKERS; do
		case $tracker in
			nn) name="NearestNeighbor" ;;
			mot) name="MultiObjectTracking" ;;
			split) name="+ BlobDetectionSplitting" ;;
			known) name="+ known number of objects" ;;
		esac
		results="$WORK/results-$scenario-$tracker.jsonl"
		if [ ! -f "$results" ] || grep -q '^unavailable' "$results"; then
			echo "| $scenario | $name | n/a | n/a | n/a | n/a | n/a | n/a |"
		elif grep -q '^failed' "$results"; then
			echo "| $scenario | $name | failed | | | | | |"
		else
			echo "| $scenario | $name | $(mean_of MOTA "$results" 3) | $(mean_of IDF1 "$results" 3) | $(mean_of HOTA "$results" 3) | $(mean_of IDSW "$results" $COUNT_DECIMALS) | $(mean_of FP "$results" $COUNT_DECIMALS) | $(mean_of FN "$results" $COUNT_DECIMALS) |"
		fi
	done
done
exit $STATUS
