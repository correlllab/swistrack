# End-to-end test of the benchmark toolchain: swistrack-synth -> SwisTrack-batch -> swistrack-eval.
# Usage: cmake -DSYNTH=... -DBATCH=... -DEVAL=... -DWORK_DIR=... [-DFRAMES=150] -P EndToEndTest.cmake
foreach(var SYNTH BATCH EVAL WORK_DIR)
	if(NOT DEFINED ${var})
		message(FATAL_ERROR "${var} is not set")
	endif()
endforeach()
if(NOT DEFINED FRAMES)
	set(FRAMES 150)
endif()

file(REMOVE_RECURSE "${WORK_DIR}")

# 1. Generate a short fish scenario
execute_process(COMMAND "${SYNTH}" --scenario fish --frames ${FRAMES} --seed 1 --out "${WORK_DIR}"
	RESULT_VARIABLE result)
if(NOT result EQUAL 0 OR NOT EXISTS "${WORK_DIR}/nn.swistrack" OR NOT EXISTS "${WORK_DIR}/groundtruth.csv")
	message(FATAL_ERROR "swistrack-synth failed (${result})")
endif()

# 2. Track with NearestNeighborTracking (SwisTrack-batch writes the CSV into the run folder "testing")
execute_process(COMMAND "${BATCH}" nn.swistrack
	WORKING_DIRECTORY "${WORK_DIR}"
	OUTPUT_VARIABLE batchOutput ERROR_VARIABLE batchOutput)
set(tracks "${WORK_DIR}/testing/nn.csv")
if(NOT EXISTS "${tracks}")
	message(FATAL_ERROR "SwisTrack-batch did not write ${tracks}. Output:\n${batchOutput}")
endif()
file(STRINGS "${tracks}" lines)
list(LENGTH lines lineCount)
if(lineCount LESS ${FRAMES})
	message(FATAL_ERROR "${tracks} has only ${lineCount} lines. SwisTrack-batch output:\n${batchOutput}")
endif()

# 3. Evaluate
execute_process(COMMAND "${EVAL}" "${WORK_DIR}/groundtruth.csv" "${tracks}" --json
	RESULT_VARIABLE result OUTPUT_VARIABLE json)
if(NOT result EQUAL 0)
	message(FATAL_ERROR "swistrack-eval failed (${result})")
endif()
message(STATUS "${json}")

string(REGEX MATCH "\"MOTA\":([-0-9.]+)" match "${json}")
set(mota "${CMAKE_MATCH_1}")
string(REGEX MATCH "\"MOTP\":([-0-9.]+)" match "${json}")
set(motp "${CMAKE_MATCH_1}")
string(REGEX MATCH "\"IDF1\":([-0-9.]+)" match "${json}")
set(idf1 "${CMAKE_MATCH_1}")
if(mota STREQUAL "" OR motp STREQUAL "" OR idf1 STREQUAL "")
	message(FATAL_ERROR "Cannot parse the swistrack-eval output: ${json}")
endif()

# Sanity bounds: detection works (MOTA), positions and frame numbers line up with the ground truth (MOTP: the
# mean match distance is ~0.2 px; an off-by-one frame or half-pixel offset would show up here)
if(NOT mota GREATER 0.5)
	message(FATAL_ERROR "MOTA ${mota} is below the sanity bound 0.5")
endif()
if(NOT motp LESS 1.0)
	message(FATAL_ERROR "MOTP ${motp} px is above the sanity bound 1 px")
endif()
if(NOT idf1 GREATER 0.3)
	message(FATAL_ERROR "IDF1 ${idf1} is below the sanity bound 0.3")
endif()
message(STATUS "MOTA ${mota}, MOTP ${motp} px, IDF1 ${idf1}: OK")
