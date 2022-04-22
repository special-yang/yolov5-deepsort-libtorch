cmake_minimum_required(VERSION 3.5)

project(module)

set(CMAKE_PREFIX_PATH "/home/yangshaopeng/project/libtorch")  #注意这里填自己解压libtorch时的路径
set(OpenCV_DIR "/home/yangshaopeng/ysp_user/opencv/build/")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
message(${OpenCV_LIBS})


# if(CMAKE_COMPILER_IS_GNUCC)
#     message("COMPILER IS GNUCC")
#         ADD_DEFINITIONS ( -std=c++11 )
# endif(CMAKE_COMPILER_IS_GNUCC)

# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -ggdb3")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# # 3. opencv
# #link_directories("/usr/local/lib")
# #include_directories("/usr/local/include/opencv4")

# # 5. 添加头文件路径
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)
# # 6. 添加源文件
# FILE(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
# message(${CMAKE_CURRENT_SOURCE_DIR})
# message(${SOURCE_FILES})
# # 7. 添加链接库
# #LINK_LIBRARIES(opencv_core opencv_highgui)

# # 9. 设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，
# #但是执行的时候会出现各种问题，比如"symbol lookup error xxxxx , undefined symbol"
# SET(ALL_SRCS ${SOURCE_FILES})
# message(${ALL_SRCS})

# # 生成so包
# ADD_LIBRARY(${PROJECT_NAME} SHARED ${ALL_SRCS})

# target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})

# # 10.add executable file，添加要编译的可执行文件
# #ADD_EXECUTABLE(${PROJECT_NAME} ${ALL_SRCS})
aux_source_directory(src TRACKING_SRCS)
add_library(tracking SHARED ${TRACKING_SRCS})
set_property(TARGET tracking PROPERTY CXX_STANDARD 14)
include(GenerateExportHeader)
GENERATE_EXPORT_HEADER(tracking)
message(${CMAKE_CURRENT_BINARY_DIR})

message( ${TRACKING_SRCS})
target_link_libraries(tracking PUBLIC ${OpenCV_LIBS} PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(tracking
        PUBLIC include ${CMAKE_CURRENT_BINARY_DIR}
        PRIVATE src)

