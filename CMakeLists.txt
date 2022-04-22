cmake_minimum_required(VERSION 3.5)

if(CMAKE_COMPILER_IS_GNUCC)
    message("COMPILER IS GNUCC")
        ADD_DEFINITIONS ( -std=c++11 )
endif(CMAKE_COMPILER_IS_GNUCC)

# Output directory structure
# set(OUTPUT_DIR "result")
# set(TARGETS_DIR_NAME "targets")
# set(TRAJ_TXT_NAME "trajectories.txt")
# set(SNAPSHOTS_DIR_NAME "snapshots")
# set(VIDEO_NAME "compressed.flv")
# configure_file(
#         "${PROJECT_SOURCE_DIR}/config.h.in"
#         "${PROJECT_BINARY_DIR}/config.h"
# )

message("project_source_dir  :"${PROJECT_SOURCE_DIR} )
message("project_binary_dir  :"${PROJECT_BINARY_DIR} )



SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -ggdb3")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/module)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/detect)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/process)

