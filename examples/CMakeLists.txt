cmake_minimum_required (VERSION 3.1)

project(CatmullClark)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
#set(CMAKE_C_FLAGS "-Os -march=native")
#set(CMAKE_C_FLAGS "-g")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(submodules/glfw)
include_directories(submodules/glfw/include)
include_directories(submodules/dj_opengl)
include_directories(..)

add_executable(obj_to_ccm obj_to_ccm.c)
add_executable(mesh_info mesh_info.c)
add_executable(subd_cpu subd_cpu.c)

add_executable(bench_cpu subd_cpu.c)
target_compile_definitions(bench_cpu PUBLIC -DFLAG_BENCH)

add_executable(subd_gpu subd_gpu.c glad/glad.c)
target_link_libraries(subd_gpu glfw)
target_compile_definitions(
    subd_gpu PUBLIC -DPATH_TO_SRC_DIRECTORY="${CMAKE_SOURCE_DIR}/"
)
IF (NOT WIN32)
    target_link_libraries(subd_gpu pthread)
ENDIF()


add_executable(bench_gpu subd_gpu.c glad/glad.c)
target_link_libraries(bench_gpu glfw)
target_compile_definitions(
    bench_gpu PUBLIC -DPATH_TO_SRC_DIRECTORY="${CMAKE_SOURCE_DIR}/" -DFLAG_BENCH
)
IF (NOT WIN32)
    target_link_libraries(bench_gpu pthread)
ENDIF()
