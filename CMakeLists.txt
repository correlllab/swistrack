cmake_minimum_required(VERSION 3.16)

project(SwisTrack VERSION 4.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SWISTRACK_BUILD_GUI "Build the SwisTrack GUI application" ON)
option(SWISTRACK_BUILD_BATCH "Build the SwisTrack-batch command line tool" ON)
option(SWISTRACK_WITH_SQLITE3 "Build the SQLite3 output component" OFF)
option(SWISTRACK_BUILD_TESTS "Build the unit tests" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui video)

set(SWISTRACK_WX_COMPONENTS xml base)
if(SWISTRACK_BUILD_GUI)
	set(SWISTRACK_WX_COMPONENTS adv net core xml base)
endif()
find_package(wxWidgets 3.0 REQUIRED COMPONENTS ${SWISTRACK_WX_COMPONENTS})
include(${wxWidgets_USE_FILE})

if(MSVC)
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES NOMINMAX)
	add_compile_options(/MP /utf-8)
endif()

# --- libtsai (camera calibration, C) ---------------------------------------
file(GLOB LIBTSAI_SOURCES CONFIGURE_DEPENDS
	3rdParty/libtsai/tsaisrc/*.c
	3rdParty/libtsai/mathsrc/*.c
	3rdParty/libtsai/mathsrc/minpack/*.c
	3rdParty/libtsai/include/*.c)
add_library(tsai STATIC ${LIBTSAI_SOURCES})
# f2c-generated K&R C: not valid in C23, which newer compilers use by default
set_target_properties(tsai PROPERTIES C_STANDARD 99 C_EXTENSIONS ON)
target_include_directories(tsai
	PUBLIC 3rdParty/libtsai/include
	PRIVATE 3rdParty/libtsai/tsaisrc 3rdParty/libtsai/mathsrc 3rdParty/libtsai/mathsrc/minpack)
if(NOT MSVC)
	target_compile_options(tsai PRIVATE -w)
	target_link_libraries(tsai PUBLIC m)
endif()

# --- SwisTrack core ---------------------------------------------------------
file(GLOB SWISTRACK_CORE_SOURCES CONFIGURE_DEPENDS Code/core/*.cpp)
add_library(SwisTrackCore STATIC ${SWISTRACK_CORE_SOURCES})
target_include_directories(SwisTrackCore PUBLIC Code/core)
target_include_directories(SwisTrackCore SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(SwisTrackCore PUBLIC tsai ${OpenCV_LIBS} ${wxWidgets_LIBRARIES})
if(SWISTRACK_WITH_SQLITE3)
	find_package(SQLite3 REQUIRED)
	target_compile_definitions(SwisTrackCore PUBLIC USE_SQLITE3)
	target_link_libraries(SwisTrackCore PUBLIC SQLite::SQLite3)
endif()

# Files the application needs at runtime (component descriptions, default config)
set(SWISTRACK_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SwisTrackEnvironment)

# --- GUI --------------------------------------------------------------------
if(SWISTRACK_BUILD_GUI)
	file(GLOB SWISTRACK_GUI_SOURCES CONFIGURE_DEPENDS Code/gui/*.cpp)
	if(WIN32)
		list(APPEND SWISTRACK_GUI_SOURCES Code/gui/MSVisualStudio/SwisTrack-GUI.rc)
	endif()

	if(APPLE)
		add_executable(SwisTrack MACOSX_BUNDLE ${SWISTRACK_GUI_SOURCES})
		set_target_properties(SwisTrack PROPERTIES
			MACOSX_BUNDLE_BUNDLE_NAME SwisTrack
			MACOSX_BUNDLE_GUI_IDENTIFIER org.swistrack.SwisTrack
			MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
			MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION})
		# Ship component descriptions inside SwisTrack.app/Contents/Resources
		add_custom_command(TARGET SwisTrack POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_directory
				${SWISTRACK_RUNTIME_DIR}/Components
				$<TARGET_BUNDLE_CONTENT_DIR:SwisTrack>/Resources/Components
			COMMAND ${CMAKE_COMMAND} -E copy
				${SWISTRACK_RUNTIME_DIR}/default.swistrack
				$<TARGET_BUNDLE_CONTENT_DIR:SwisTrack>/Resources/default.swistrack)
	else()
		add_executable(SwisTrack WIN32 ${SWISTRACK_GUI_SOURCES})
	endif()
	target_include_directories(SwisTrack PRIVATE Code/gui)
	target_link_libraries(SwisTrack PRIVATE SwisTrackCore)
endif()

# --- Batch ------------------------------------------------------------------
if(SWISTRACK_BUILD_BATCH)
	file(GLOB SWISTRACK_BATCH_SOURCES CONFIGURE_DEPENDS Code/batch/*.cpp)
	add_executable(SwisTrack-batch ${SWISTRACK_BATCH_SOURCES})
	target_link_libraries(SwisTrack-batch PRIVATE SwisTrackCore)
endif()

# --- Copy runtime files next to the executables (build tree) -----------------
function(swistrack_copy_runtime_files target)
	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_directory
			${SWISTRACK_RUNTIME_DIR}/Components $<TARGET_FILE_DIR:${target}>/Components
		COMMAND ${CMAKE_COMMAND} -E copy
			${SWISTRACK_RUNTIME_DIR}/default.swistrack $<TARGET_FILE_DIR:${target}>/default.swistrack)
endfunction()
if(SWISTRACK_BUILD_GUI AND NOT APPLE)
	swistrack_copy_runtime_files(SwisTrack)
endif()
if(SWISTRACK_BUILD_BATCH)
	swistrack_copy_runtime_files(SwisTrack-batch)
endif()

# --- Tests ------------------------------------------------------------------
if(SWISTRACK_BUILD_TESTS)
	enable_testing()
	add_executable(TestLinearAssignment Code/tests/TestLinearAssignment.cpp Code/core/LinearAssignment.cpp)
	target_include_directories(TestLinearAssignment PRIVATE Code/core)
	add_test(NAME LinearAssignment COMMAND TestLinearAssignment)
	add_executable(TestMultiObjectTracker Code/tests/TestMultiObjectTracker.cpp Code/core/MultiObjectTracker.cpp Code/core/LinearAssignment.cpp)
	target_include_directories(TestMultiObjectTracker PRIVATE Code/core)
	target_include_directories(TestMultiObjectTracker SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(TestMultiObjectTracker PRIVATE opencv_core)
	add_test(NAME MultiObjectTracker COMMAND TestMultiObjectTracker)
	add_executable(TestBlobSplitter Code/tests/TestBlobSplitter.cpp Code/core/BlobSplitter.cpp)
	target_include_directories(TestBlobSplitter PRIVATE Code/core)
	target_include_directories(TestBlobSplitter SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
	target_link_libraries(TestBlobSplitter PRIVATE opencv_core opencv_imgproc)
	add_test(NAME BlobSplitter COMMAND TestBlobSplitter)
endif()

# --- Install ----------------------------------------------------------------
include(GNUInstallDirs)
if(APPLE)
	if(SWISTRACK_BUILD_GUI)
		install(TARGETS SwisTrack BUNDLE DESTINATION .)
	endif()
	if(SWISTRACK_BUILD_BATCH)
		install(TARGETS SwisTrack-batch RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
		install(DIRECTORY ${SWISTRACK_RUNTIME_DIR}/Components DESTINATION ${CMAKE_INSTALL_DATADIR}/swistrack)
		install(FILES ${SWISTRACK_RUNTIME_DIR}/default.swistrack DESTINATION ${CMAKE_INSTALL_DATADIR}/swistrack)
	endif()
elseif(WIN32)
	set(_targets)
	if(SWISTRACK_BUILD_GUI)
		list(APPEND _targets SwisTrack)
	endif()
	if(SWISTRACK_BUILD_BATCH)
		list(APPEND _targets SwisTrack-batch)
	endif()
	install(TARGETS ${_targets} RUNTIME DESTINATION .)
	install(DIRECTORY ${SWISTRACK_RUNTIME_DIR}/Components DESTINATION .)
	install(FILES ${SWISTRACK_RUNTIME_DIR}/default.swistrack DESTINATION .)
else()
	set(_targets)
	if(SWISTRACK_BUILD_GUI)
		list(APPEND _targets SwisTrack)
	endif()
	if(SWISTRACK_BUILD_BATCH)
		list(APPEND _targets SwisTrack-batch)
	endif()
	install(TARGETS ${_targets} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
	install(DIRECTORY ${SWISTRACK_RUNTIME_DIR}/Components DESTINATION ${CMAKE_INSTALL_DATADIR}/swistrack)
	install(FILES ${SWISTRACK_RUNTIME_DIR}/default.swistrack DESTINATION ${CMAKE_INSTALL_DATADIR}/swistrack)
endif()
