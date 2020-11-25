cmake_minimum_required(VERSION 2.8)
project(APAP LANGUAGES CUDA CXX)

SET(GCC_COVERAGE_COMPILE_FLAGS "-pg")
SET(GCC_COVERAGE_LINK_FLAGS    "-pg")

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread -D WITH_LIBV4L=ON ")
# SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-lpthread -lvisionworks -lovx -lnvx ")
# SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGE} "-lgstpbutils-1.0 -lgstapp-1.0 -lgstbase-1.0 -lgstreamer-1.o -lgobject-2.0 -lglib-2.0")
# SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGE} "/usr/share/visionworks/sorces/3rdparty/glfw3/libs/libglfw3.a")
# SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGE} "-lXrandr -lXi -lXxf86vm -lX11 -lGL")


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 3.4 REQUIRED)
find_package(OpenCL REQUIRED)
# find_package(GLEW REQUIRED)
# find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
# find_package(DevIL REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(VisionWorks REQUIRED)

# find_package(jetson-utils)
# find_package(jetson-inference)
# find_package(OpenMP REQUIRED)

# set(VLFEAT_INCLUDE ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/include)
# set(VLFEAT_LIB ${PROJECT_SOURCE_DIR}/thirdparty/vlfeat/lib/libvl.so)
# set(SIFTGPU_INCLUDE ${PROJECT_SOURCE_DIR}/thirdparty/SiftGPU/include)
# set(SIFTGPU_LIB ${PROJECT_SOURCE_DIR}/thirdparty/SiftGPU/lib/libsiftgpu.so)
# set(MIVISIONX_INCLUDE /opt/rocm/mivisionx/include)
set(VW_INCLUDE /usr/share/visionworks/sources/nvxio/include)

add_definitions(-std=c++11)

include_directories(
  # ${GLEW_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIRS}
  # ${GLUT_INCLUDE_DIRS}
  # ${IL_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${VW_INCLUDE}
  # ${VLFEAT_INCLUDE}
  # ${SIFTGPU_INCLUDE}
  # ${MIVISIONX_INCLUDE}
)

link_directories(
  "/usr/local/cuda/targets/aarch64-linux/lib"
  "/sr/share/visionworks/sources/libs/aarch64/linux/release"
)

# link_directories(/opt/rocm/mivisionx/lib)

add_executable(APAP ${PROJECT_SOURCE_DIR}/src/APAP.cpp
                    ${PROJECT_SOURCE_DIR}/src/Math.cpp
                    ${PROJECT_SOURCE_DIR}/src/CVUtility.cpp
                    # ${PROJECT_SOURCE_DIR}/src/SiftGPUWrapper.cpp
                    # ${PROJECT_SOURCE_DIR}/src/VLFeatSiftWrapper.cpp
                    # ${PROJECT_SOURCE_DIR}/src/SiftROCm.cpp
)

target_link_libraries(APAP 
  # ${GLEW_LIBRARIES}
  # ${GLUT_LIBRARY}
  ${OPENGL_LIBRARIES}
  # ${IL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${EIGEN3_LIBS}
  # ${VisionWorks_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDA_CUSOLVER_LIBRARIES}
  # ${VLFEAT_LIB}
  # ${SIFTGPU_LIB}
  # openvx 
  # vx_opencv
  # OpenMP::OpenMP_CXX
)
