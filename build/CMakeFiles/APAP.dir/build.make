# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/shane/stitching/AsProjectiveAsPossible

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shane/stitching/AsProjectiveAsPossible/build

# Include any dependencies generated for this target.
include CMakeFiles/APAP.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/APAP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/APAP.dir/flags.make

CMakeFiles/APAP.dir/src/APAP.cpp.o: CMakeFiles/APAP.dir/flags.make
CMakeFiles/APAP.dir/src/APAP.cpp.o: ../src/APAP.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shane/stitching/AsProjectiveAsPossible/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/APAP.dir/src/APAP.cpp.o"
	/bin/x86_64-linux-gnu-g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/APAP.dir/src/APAP.cpp.o -c /home/shane/stitching/AsProjectiveAsPossible/src/APAP.cpp

CMakeFiles/APAP.dir/src/APAP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/APAP.dir/src/APAP.cpp.i"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shane/stitching/AsProjectiveAsPossible/src/APAP.cpp > CMakeFiles/APAP.dir/src/APAP.cpp.i

CMakeFiles/APAP.dir/src/APAP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/APAP.dir/src/APAP.cpp.s"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shane/stitching/AsProjectiveAsPossible/src/APAP.cpp -o CMakeFiles/APAP.dir/src/APAP.cpp.s

CMakeFiles/APAP.dir/src/Math.cpp.o: CMakeFiles/APAP.dir/flags.make
CMakeFiles/APAP.dir/src/Math.cpp.o: ../src/Math.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shane/stitching/AsProjectiveAsPossible/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/APAP.dir/src/Math.cpp.o"
	/bin/x86_64-linux-gnu-g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/APAP.dir/src/Math.cpp.o -c /home/shane/stitching/AsProjectiveAsPossible/src/Math.cpp

CMakeFiles/APAP.dir/src/Math.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/APAP.dir/src/Math.cpp.i"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shane/stitching/AsProjectiveAsPossible/src/Math.cpp > CMakeFiles/APAP.dir/src/Math.cpp.i

CMakeFiles/APAP.dir/src/Math.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/APAP.dir/src/Math.cpp.s"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shane/stitching/AsProjectiveAsPossible/src/Math.cpp -o CMakeFiles/APAP.dir/src/Math.cpp.s

CMakeFiles/APAP.dir/src/CVUtility.cpp.o: CMakeFiles/APAP.dir/flags.make
CMakeFiles/APAP.dir/src/CVUtility.cpp.o: ../src/CVUtility.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shane/stitching/AsProjectiveAsPossible/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/APAP.dir/src/CVUtility.cpp.o"
	/bin/x86_64-linux-gnu-g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/APAP.dir/src/CVUtility.cpp.o -c /home/shane/stitching/AsProjectiveAsPossible/src/CVUtility.cpp

CMakeFiles/APAP.dir/src/CVUtility.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/APAP.dir/src/CVUtility.cpp.i"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shane/stitching/AsProjectiveAsPossible/src/CVUtility.cpp > CMakeFiles/APAP.dir/src/CVUtility.cpp.i

CMakeFiles/APAP.dir/src/CVUtility.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/APAP.dir/src/CVUtility.cpp.s"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shane/stitching/AsProjectiveAsPossible/src/CVUtility.cpp -o CMakeFiles/APAP.dir/src/CVUtility.cpp.s

CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.o: CMakeFiles/APAP.dir/flags.make
CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.o: ../src/SiftGPUWrapper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shane/stitching/AsProjectiveAsPossible/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.o"
	/bin/x86_64-linux-gnu-g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.o -c /home/shane/stitching/AsProjectiveAsPossible/src/SiftGPUWrapper.cpp

CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.i"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shane/stitching/AsProjectiveAsPossible/src/SiftGPUWrapper.cpp > CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.i

CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.s"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shane/stitching/AsProjectiveAsPossible/src/SiftGPUWrapper.cpp -o CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.s

CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.o: CMakeFiles/APAP.dir/flags.make
CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.o: ../src/VLFeatSiftWrapper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shane/stitching/AsProjectiveAsPossible/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.o"
	/bin/x86_64-linux-gnu-g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.o -c /home/shane/stitching/AsProjectiveAsPossible/src/VLFeatSiftWrapper.cpp

CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.i"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shane/stitching/AsProjectiveAsPossible/src/VLFeatSiftWrapper.cpp > CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.i

CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.s"
	/bin/x86_64-linux-gnu-g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shane/stitching/AsProjectiveAsPossible/src/VLFeatSiftWrapper.cpp -o CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.s

# Object files for target APAP
APAP_OBJECTS = \
"CMakeFiles/APAP.dir/src/APAP.cpp.o" \
"CMakeFiles/APAP.dir/src/Math.cpp.o" \
"CMakeFiles/APAP.dir/src/CVUtility.cpp.o" \
"CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.o" \
"CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.o"

# External object files for target APAP
APAP_EXTERNAL_OBJECTS =

APAP: CMakeFiles/APAP.dir/src/APAP.cpp.o
APAP: CMakeFiles/APAP.dir/src/Math.cpp.o
APAP: CMakeFiles/APAP.dir/src/CVUtility.cpp.o
APAP: CMakeFiles/APAP.dir/src/SiftGPUWrapper.cpp.o
APAP: CMakeFiles/APAP.dir/src/VLFeatSiftWrapper.cpp.o
APAP: CMakeFiles/APAP.dir/build.make
APAP: /usr/lib/x86_64-linux-gnu/libGLEW.so
APAP: /usr/lib/x86_64-linux-gnu/libglut.so
APAP: /usr/lib/x86_64-linux-gnu/libXmu.so
APAP: /usr/lib/x86_64-linux-gnu/libXi.so
APAP: /usr/lib/x86_64-linux-gnu/libGL.so
APAP: /usr/lib/x86_64-linux-gnu/libGLU.so
APAP: /usr/lib/x86_64-linux-gnu/libIL.so
APAP: /usr/local/lib/libopencv_stitching.so.3.4.12
APAP: /usr/local/lib/libopencv_superres.so.3.4.12
APAP: /usr/local/lib/libopencv_videostab.so.3.4.12
APAP: /usr/local/lib/libopencv_aruco.so.3.4.12
APAP: /usr/local/lib/libopencv_bgsegm.so.3.4.12
APAP: /usr/local/lib/libopencv_bioinspired.so.3.4.12
APAP: /usr/local/lib/libopencv_ccalib.so.3.4.12
APAP: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.12
APAP: /usr/local/lib/libopencv_dpm.so.3.4.12
APAP: /usr/local/lib/libopencv_face.so.3.4.12
APAP: /usr/local/lib/libopencv_freetype.so.3.4.12
APAP: /usr/local/lib/libopencv_fuzzy.so.3.4.12
APAP: /usr/local/lib/libopencv_hdf.so.3.4.12
APAP: /usr/local/lib/libopencv_hfs.so.3.4.12
APAP: /usr/local/lib/libopencv_img_hash.so.3.4.12
APAP: /usr/local/lib/libopencv_line_descriptor.so.3.4.12
APAP: /usr/local/lib/libopencv_optflow.so.3.4.12
APAP: /usr/local/lib/libopencv_reg.so.3.4.12
APAP: /usr/local/lib/libopencv_rgbd.so.3.4.12
APAP: /usr/local/lib/libopencv_saliency.so.3.4.12
APAP: /usr/local/lib/libopencv_stereo.so.3.4.12
APAP: /usr/local/lib/libopencv_structured_light.so.3.4.12
APAP: /usr/local/lib/libopencv_surface_matching.so.3.4.12
APAP: /usr/local/lib/libopencv_tracking.so.3.4.12
APAP: /usr/local/lib/libopencv_xfeatures2d.so.3.4.12
APAP: /usr/local/lib/libopencv_ximgproc.so.3.4.12
APAP: /usr/local/lib/libopencv_xobjdetect.so.3.4.12
APAP: /usr/local/lib/libopencv_xphoto.so.3.4.12
APAP: ../thirdparty/vlfeat/lib/libvl.so
APAP: /usr/local/lib/libopencv_shape.so.3.4.12
APAP: /usr/local/lib/libopencv_highgui.so.3.4.12
APAP: /usr/local/lib/libopencv_videoio.so.3.4.12
APAP: /usr/local/lib/libopencv_viz.so.3.4.12
APAP: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.12
APAP: /usr/local/lib/libopencv_video.so.3.4.12
APAP: /usr/local/lib/libopencv_datasets.so.3.4.12
APAP: /usr/local/lib/libopencv_plot.so.3.4.12
APAP: /usr/local/lib/libopencv_text.so.3.4.12
APAP: /usr/local/lib/libopencv_dnn.so.3.4.12
APAP: /usr/local/lib/libopencv_ml.so.3.4.12
APAP: /usr/local/lib/libopencv_imgcodecs.so.3.4.12
APAP: /usr/local/lib/libopencv_objdetect.so.3.4.12
APAP: /usr/local/lib/libopencv_calib3d.so.3.4.12
APAP: /usr/local/lib/libopencv_features2d.so.3.4.12
APAP: /usr/local/lib/libopencv_flann.so.3.4.12
APAP: /usr/local/lib/libopencv_photo.so.3.4.12
APAP: /usr/local/lib/libopencv_imgproc.so.3.4.12
APAP: /usr/local/lib/libopencv_core.so.3.4.12
APAP: CMakeFiles/APAP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shane/stitching/AsProjectiveAsPossible/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable APAP"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/APAP.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/APAP.dir/build: APAP

.PHONY : CMakeFiles/APAP.dir/build

CMakeFiles/APAP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/APAP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/APAP.dir/clean

CMakeFiles/APAP.dir/depend:
	cd /home/shane/stitching/AsProjectiveAsPossible/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shane/stitching/AsProjectiveAsPossible /home/shane/stitching/AsProjectiveAsPossible /home/shane/stitching/AsProjectiveAsPossible/build /home/shane/stitching/AsProjectiveAsPossible/build /home/shane/stitching/AsProjectiveAsPossible/build/CMakeFiles/APAP.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/APAP.dir/depend

