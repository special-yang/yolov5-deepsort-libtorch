# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

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
CMAKE_COMMAND = /home/yangshaopeng/anaconda3/bin/cmake

# The command to remove a file.
RM = /home/yangshaopeng/anaconda3/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yangshaopeng/project/count_cplus/module

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yangshaopeng/project/count_cplus/module/build

# Include any dependencies generated for this target.
include CMakeFiles/tracking.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tracking.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tracking.dir/flags.make

CMakeFiles/tracking.dir/src/DeepSORT.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/DeepSORT.cpp.o: ../src/DeepSORT.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yangshaopeng/project/count_cplus/module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tracking.dir/src/DeepSORT.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/DeepSORT.cpp.o -c /home/yangshaopeng/project/count_cplus/module/src/DeepSORT.cpp

CMakeFiles/tracking.dir/src/DeepSORT.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/DeepSORT.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yangshaopeng/project/count_cplus/module/src/DeepSORT.cpp > CMakeFiles/tracking.dir/src/DeepSORT.cpp.i

CMakeFiles/tracking.dir/src/DeepSORT.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/DeepSORT.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yangshaopeng/project/count_cplus/module/src/DeepSORT.cpp -o CMakeFiles/tracking.dir/src/DeepSORT.cpp.s

CMakeFiles/tracking.dir/src/Hungarian.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/Hungarian.cpp.o: ../src/Hungarian.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yangshaopeng/project/count_cplus/module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/tracking.dir/src/Hungarian.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/Hungarian.cpp.o -c /home/yangshaopeng/project/count_cplus/module/src/Hungarian.cpp

CMakeFiles/tracking.dir/src/Hungarian.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/Hungarian.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yangshaopeng/project/count_cplus/module/src/Hungarian.cpp > CMakeFiles/tracking.dir/src/Hungarian.cpp.i

CMakeFiles/tracking.dir/src/Hungarian.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/Hungarian.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yangshaopeng/project/count_cplus/module/src/Hungarian.cpp -o CMakeFiles/tracking.dir/src/Hungarian.cpp.s

CMakeFiles/tracking.dir/src/KalmanTracker.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/KalmanTracker.cpp.o: ../src/KalmanTracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yangshaopeng/project/count_cplus/module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/tracking.dir/src/KalmanTracker.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/KalmanTracker.cpp.o -c /home/yangshaopeng/project/count_cplus/module/src/KalmanTracker.cpp

CMakeFiles/tracking.dir/src/KalmanTracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/KalmanTracker.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yangshaopeng/project/count_cplus/module/src/KalmanTracker.cpp > CMakeFiles/tracking.dir/src/KalmanTracker.cpp.i

CMakeFiles/tracking.dir/src/KalmanTracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/KalmanTracker.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yangshaopeng/project/count_cplus/module/src/KalmanTracker.cpp -o CMakeFiles/tracking.dir/src/KalmanTracker.cpp.s

CMakeFiles/tracking.dir/src/SORT.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/SORT.cpp.o: ../src/SORT.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yangshaopeng/project/count_cplus/module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/tracking.dir/src/SORT.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/SORT.cpp.o -c /home/yangshaopeng/project/count_cplus/module/src/SORT.cpp

CMakeFiles/tracking.dir/src/SORT.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/SORT.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yangshaopeng/project/count_cplus/module/src/SORT.cpp > CMakeFiles/tracking.dir/src/SORT.cpp.i

CMakeFiles/tracking.dir/src/SORT.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/SORT.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yangshaopeng/project/count_cplus/module/src/SORT.cpp -o CMakeFiles/tracking.dir/src/SORT.cpp.s

CMakeFiles/tracking.dir/src/TrackerManager.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/TrackerManager.cpp.o: ../src/TrackerManager.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yangshaopeng/project/count_cplus/module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/tracking.dir/src/TrackerManager.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/TrackerManager.cpp.o -c /home/yangshaopeng/project/count_cplus/module/src/TrackerManager.cpp

CMakeFiles/tracking.dir/src/TrackerManager.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/TrackerManager.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yangshaopeng/project/count_cplus/module/src/TrackerManager.cpp > CMakeFiles/tracking.dir/src/TrackerManager.cpp.i

CMakeFiles/tracking.dir/src/TrackerManager.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/TrackerManager.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yangshaopeng/project/count_cplus/module/src/TrackerManager.cpp -o CMakeFiles/tracking.dir/src/TrackerManager.cpp.s

CMakeFiles/tracking.dir/src/extra.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/extra.cpp.o: ../src/extra.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yangshaopeng/project/count_cplus/module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/tracking.dir/src/extra.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/extra.cpp.o -c /home/yangshaopeng/project/count_cplus/module/src/extra.cpp

CMakeFiles/tracking.dir/src/extra.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/extra.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yangshaopeng/project/count_cplus/module/src/extra.cpp > CMakeFiles/tracking.dir/src/extra.cpp.i

CMakeFiles/tracking.dir/src/extra.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/extra.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yangshaopeng/project/count_cplus/module/src/extra.cpp -o CMakeFiles/tracking.dir/src/extra.cpp.s

CMakeFiles/tracking.dir/src/nn_matching.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/nn_matching.cpp.o: ../src/nn_matching.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yangshaopeng/project/count_cplus/module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/tracking.dir/src/nn_matching.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/nn_matching.cpp.o -c /home/yangshaopeng/project/count_cplus/module/src/nn_matching.cpp

CMakeFiles/tracking.dir/src/nn_matching.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/nn_matching.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yangshaopeng/project/count_cplus/module/src/nn_matching.cpp > CMakeFiles/tracking.dir/src/nn_matching.cpp.i

CMakeFiles/tracking.dir/src/nn_matching.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/nn_matching.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yangshaopeng/project/count_cplus/module/src/nn_matching.cpp -o CMakeFiles/tracking.dir/src/nn_matching.cpp.s

# Object files for target tracking
tracking_OBJECTS = \
"CMakeFiles/tracking.dir/src/DeepSORT.cpp.o" \
"CMakeFiles/tracking.dir/src/Hungarian.cpp.o" \
"CMakeFiles/tracking.dir/src/KalmanTracker.cpp.o" \
"CMakeFiles/tracking.dir/src/SORT.cpp.o" \
"CMakeFiles/tracking.dir/src/TrackerManager.cpp.o" \
"CMakeFiles/tracking.dir/src/extra.cpp.o" \
"CMakeFiles/tracking.dir/src/nn_matching.cpp.o"

# External object files for target tracking
tracking_EXTERNAL_OBJECTS =

libtracking.so: CMakeFiles/tracking.dir/src/DeepSORT.cpp.o
libtracking.so: CMakeFiles/tracking.dir/src/Hungarian.cpp.o
libtracking.so: CMakeFiles/tracking.dir/src/KalmanTracker.cpp.o
libtracking.so: CMakeFiles/tracking.dir/src/SORT.cpp.o
libtracking.so: CMakeFiles/tracking.dir/src/TrackerManager.cpp.o
libtracking.so: CMakeFiles/tracking.dir/src/extra.cpp.o
libtracking.so: CMakeFiles/tracking.dir/src/nn_matching.cpp.o
libtracking.so: CMakeFiles/tracking.dir/build.make
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_gapi.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_highgui.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_ml.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_objdetect.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_photo.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_stitching.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_video.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_videoio.so.4.5.2
libtracking.so: /home/yangshaopeng/project/libtorch/lib/libtorch.so
libtracking.so: /home/yangshaopeng/project/libtorch/lib/libc10.so
libtracking.so: /home/yangshaopeng/ysp_user/cuda10_1/lib64/stubs/libcuda.so
libtracking.so: /home/yangshaopeng/ysp_user/cuda10_1/lib64/libnvrtc.so
libtracking.so: /home/yangshaopeng/ysp_user/cuda10_1/lib64/libnvToolsExt.so
libtracking.so: /home/yangshaopeng/ysp_user/cuda10_1/lib64/libcudart.so
libtracking.so: /home/yangshaopeng/project/libtorch/lib/libc10_cuda.so
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_dnn.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_imgcodecs.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_calib3d.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_features2d.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_flann.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_imgproc.so.4.5.2
libtracking.so: /home/yangshaopeng/ysp_user/opencv/build/lib/libopencv_core.so.4.5.2
libtracking.so: /home/yangshaopeng/project/libtorch/lib/libc10_cuda.so
libtracking.so: /home/yangshaopeng/project/libtorch/lib/libc10.so
libtracking.so: /home/yangshaopeng/ysp_user/cuda10_1/lib64/libcufft.so
libtracking.so: /home/yangshaopeng/ysp_user/cuda10_1/lib64/libcurand.so
libtracking.so: /home/yangshaopeng/ysp_user/cuda10_1/lib64/libcublas.so
libtracking.so: /home/yangshaopeng/ysp_user/cuda10_1/lib64/libcudnn.so
libtracking.so: /home/yangshaopeng/ysp_user/cuda10_1/lib64/libnvToolsExt.so
libtracking.so: /home/yangshaopeng/ysp_user/cuda10_1/lib64/libcudart.so
libtracking.so: CMakeFiles/tracking.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yangshaopeng/project/count_cplus/module/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX shared library libtracking.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tracking.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tracking.dir/build: libtracking.so

.PHONY : CMakeFiles/tracking.dir/build

CMakeFiles/tracking.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tracking.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tracking.dir/clean

CMakeFiles/tracking.dir/depend:
	cd /home/yangshaopeng/project/count_cplus/module/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yangshaopeng/project/count_cplus/module /home/yangshaopeng/project/count_cplus/module /home/yangshaopeng/project/count_cplus/module/build /home/yangshaopeng/project/count_cplus/module/build /home/yangshaopeng/project/count_cplus/module/build/CMakeFiles/tracking.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tracking.dir/depend

