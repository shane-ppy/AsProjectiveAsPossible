# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pi/stitching/AsProjectiveAsPossible

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pi/stitching/AsProjectiveAsPossible/build

# Include any dependencies generated for this target.
include CMakeFiles/APAP.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/APAP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/APAP.dir/flags.make

CMakeFiles/APAP.dir/src/APAP.cpp.o: CMakeFiles/APAP.dir/flags.make
CMakeFiles/APAP.dir/src/APAP.cpp.o: ../src/APAP.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/stitching/AsProjectiveAsPossible/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/APAP.dir/src/APAP.cpp.o"
	/usr/bin/aarch64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/APAP.dir/src/APAP.cpp.o -c /home/pi/stitching/AsProjectiveAsPossible/src/APAP.cpp

CMakeFiles/APAP.dir/src/APAP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/APAP.dir/src/APAP.cpp.i"
	/usr/bin/aarch64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/stitching/AsProjectiveAsPossible/src/APAP.cpp > CMakeFiles/APAP.dir/src/APAP.cpp.i

CMakeFiles/APAP.dir/src/APAP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/APAP.dir/src/APAP.cpp.s"
	/usr/bin/aarch64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/stitching/AsProjectiveAsPossible/src/APAP.cpp -o CMakeFiles/APAP.dir/src/APAP.cpp.s

CMakeFiles/APAP.dir/src/APAP.cpp.o.requires:

.PHONY : CMakeFiles/APAP.dir/src/APAP.cpp.o.requires

CMakeFiles/APAP.dir/src/APAP.cpp.o.provides: CMakeFiles/APAP.dir/src/APAP.cpp.o.requires
	$(MAKE) -f CMakeFiles/APAP.dir/build.make CMakeFiles/APAP.dir/src/APAP.cpp.o.provides.build
.PHONY : CMakeFiles/APAP.dir/src/APAP.cpp.o.provides

CMakeFiles/APAP.dir/src/APAP.cpp.o.provides.build: CMakeFiles/APAP.dir/src/APAP.cpp.o


CMakeFiles/APAP.dir/src/Math.cpp.o: CMakeFiles/APAP.dir/flags.make
CMakeFiles/APAP.dir/src/Math.cpp.o: ../src/Math.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/stitching/AsProjectiveAsPossible/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/APAP.dir/src/Math.cpp.o"
	/usr/bin/aarch64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/APAP.dir/src/Math.cpp.o -c /home/pi/stitching/AsProjectiveAsPossible/src/Math.cpp

CMakeFiles/APAP.dir/src/Math.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/APAP.dir/src/Math.cpp.i"
	/usr/bin/aarch64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/stitching/AsProjectiveAsPossible/src/Math.cpp > CMakeFiles/APAP.dir/src/Math.cpp.i

CMakeFiles/APAP.dir/src/Math.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/APAP.dir/src/Math.cpp.s"
	/usr/bin/aarch64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/stitching/AsProjectiveAsPossible/src/Math.cpp -o CMakeFiles/APAP.dir/src/Math.cpp.s

CMakeFiles/APAP.dir/src/Math.cpp.o.requires:

.PHONY : CMakeFiles/APAP.dir/src/Math.cpp.o.requires

CMakeFiles/APAP.dir/src/Math.cpp.o.provides: CMakeFiles/APAP.dir/src/Math.cpp.o.requires
	$(MAKE) -f CMakeFiles/APAP.dir/build.make CMakeFiles/APAP.dir/src/Math.cpp.o.provides.build
.PHONY : CMakeFiles/APAP.dir/src/Math.cpp.o.provides

CMakeFiles/APAP.dir/src/Math.cpp.o.provides.build: CMakeFiles/APAP.dir/src/Math.cpp.o


CMakeFiles/APAP.dir/src/CVUtility.cpp.o: CMakeFiles/APAP.dir/flags.make
CMakeFiles/APAP.dir/src/CVUtility.cpp.o: ../src/CVUtility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/stitching/AsProjectiveAsPossible/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/APAP.dir/src/CVUtility.cpp.o"
	/usr/bin/aarch64-linux-gnu-g++-7  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/APAP.dir/src/CVUtility.cpp.o -c /home/pi/stitching/AsProjectiveAsPossible/src/CVUtility.cpp

CMakeFiles/APAP.dir/src/CVUtility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/APAP.dir/src/CVUtility.cpp.i"
	/usr/bin/aarch64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/stitching/AsProjectiveAsPossible/src/CVUtility.cpp > CMakeFiles/APAP.dir/src/CVUtility.cpp.i

CMakeFiles/APAP.dir/src/CVUtility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/APAP.dir/src/CVUtility.cpp.s"
	/usr/bin/aarch64-linux-gnu-g++-7 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/stitching/AsProjectiveAsPossible/src/CVUtility.cpp -o CMakeFiles/APAP.dir/src/CVUtility.cpp.s

CMakeFiles/APAP.dir/src/CVUtility.cpp.o.requires:

.PHONY : CMakeFiles/APAP.dir/src/CVUtility.cpp.o.requires

CMakeFiles/APAP.dir/src/CVUtility.cpp.o.provides: CMakeFiles/APAP.dir/src/CVUtility.cpp.o.requires
	$(MAKE) -f CMakeFiles/APAP.dir/build.make CMakeFiles/APAP.dir/src/CVUtility.cpp.o.provides.build
.PHONY : CMakeFiles/APAP.dir/src/CVUtility.cpp.o.provides

CMakeFiles/APAP.dir/src/CVUtility.cpp.o.provides.build: CMakeFiles/APAP.dir/src/CVUtility.cpp.o


# Object files for target APAP
APAP_OBJECTS = \
"CMakeFiles/APAP.dir/src/APAP.cpp.o" \
"CMakeFiles/APAP.dir/src/Math.cpp.o" \
"CMakeFiles/APAP.dir/src/CVUtility.cpp.o"

# External object files for target APAP
APAP_EXTERNAL_OBJECTS =

APAP: CMakeFiles/APAP.dir/src/APAP.cpp.o
APAP: CMakeFiles/APAP.dir/src/Math.cpp.o
APAP: CMakeFiles/APAP.dir/src/CVUtility.cpp.o
APAP: CMakeFiles/APAP.dir/build.make
APAP: /usr/lib/aarch64-linux-gnu/libGL.so
APAP: /usr/lib/aarch64-linux-gnu/libGLU.so
APAP: /usr/local/lib/libopencv_cudabgsegm.so.3.4.1
APAP: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.1
APAP: /usr/local/lib/libopencv_cudastereo.so.3.4.1
APAP: /usr/local/lib/libopencv_stitching.so.3.4.1
APAP: /usr/local/lib/libopencv_superres.so.3.4.1
APAP: /usr/local/lib/libopencv_videostab.so.3.4.1
APAP: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.1
APAP: /usr/local/lib/libopencv_face.so.3.4.1
APAP: /usr/local/lib/libopencv_img_hash.so.3.4.1
APAP: /usr/local/lib/libopencv_reg.so.3.4.1
APAP: /usr/local/lib/libopencv_stereo.so.3.4.1
APAP: /usr/local/lib/libopencv_surface_matching.so.3.4.1
APAP: /usr/local/lib/libopencv_tracking.so.3.4.1
APAP: /usr/local/lib/libopencv_xfeatures2d.so.3.4.1
APAP: /usr/local/lib/libopencv_ximgproc.so.3.4.1
APAP: /usr/local/cuda-10.2/lib64/libcudart_static.a
APAP: /usr/lib/aarch64-linux-gnu/librt.so
APAP: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.1
APAP: /usr/local/lib/libopencv_shape.so.3.4.1
APAP: /usr/local/lib/libopencv_cudacodec.so.3.4.1
APAP: /usr/local/lib/libopencv_cudaoptflow.so.3.4.1
APAP: /usr/local/lib/libopencv_cudalegacy.so.3.4.1
APAP: /usr/local/lib/libopencv_cudawarping.so.3.4.1
APAP: /usr/local/lib/libopencv_objdetect.so.3.4.1
APAP: /usr/local/lib/libopencv_photo.so.3.4.1
APAP: /usr/local/lib/libopencv_cudaimgproc.so.3.4.1
APAP: /usr/local/lib/libopencv_cudafilters.so.3.4.1
APAP: /usr/local/lib/libopencv_cudaarithm.so.3.4.1
APAP: /usr/local/lib/libopencv_video.so.3.4.1
APAP: /usr/local/lib/libopencv_datasets.so.3.4.1
APAP: /usr/local/lib/libopencv_plot.so.3.4.1
APAP: /usr/local/lib/libopencv_text.so.3.4.1
APAP: /usr/local/lib/libopencv_dnn.so.3.4.1
APAP: /usr/local/lib/libopencv_ml.so.3.4.1
APAP: /usr/local/lib/libopencv_calib3d.so.3.4.1
APAP: /usr/local/lib/libopencv_features2d.so.3.4.1
APAP: /usr/local/lib/libopencv_flann.so.3.4.1
APAP: /usr/local/lib/libopencv_highgui.so.3.4.1
APAP: /usr/local/lib/libopencv_videoio.so.3.4.1
APAP: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
APAP: /usr/local/lib/libopencv_imgproc.so.3.4.1
APAP: /usr/local/lib/libopencv_core.so.3.4.1
APAP: /usr/local/lib/libopencv_cudev.so.3.4.1
APAP: CMakeFiles/APAP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pi/stitching/AsProjectiveAsPossible/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable APAP"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/APAP.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/APAP.dir/build: APAP

.PHONY : CMakeFiles/APAP.dir/build

CMakeFiles/APAP.dir/requires: CMakeFiles/APAP.dir/src/APAP.cpp.o.requires
CMakeFiles/APAP.dir/requires: CMakeFiles/APAP.dir/src/Math.cpp.o.requires
CMakeFiles/APAP.dir/requires: CMakeFiles/APAP.dir/src/CVUtility.cpp.o.requires

.PHONY : CMakeFiles/APAP.dir/requires

CMakeFiles/APAP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/APAP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/APAP.dir/clean

CMakeFiles/APAP.dir/depend:
	cd /home/pi/stitching/AsProjectiveAsPossible/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pi/stitching/AsProjectiveAsPossible /home/pi/stitching/AsProjectiveAsPossible /home/pi/stitching/AsProjectiveAsPossible/build /home/pi/stitching/AsProjectiveAsPossible/build /home/pi/stitching/AsProjectiveAsPossible/build/CMakeFiles/APAP.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/APAP.dir/depend

