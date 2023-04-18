cmake_minimum_required(VERSION 3.14)
project(CatmullClarkRewrite)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(OpenMP REQUIRED)

include_directories(..)
add_executable(openmp subd_rewrite.cpp ../CatmullClark.cpp ../Mesh.cpp ../Utilities.h)
target_link_libraries(openmp PRIVATE OpenMP::OpenMP_CXX)
