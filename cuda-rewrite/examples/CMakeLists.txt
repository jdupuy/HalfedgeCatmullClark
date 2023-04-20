cmake_minimum_required(VERSION 3.14)
enable_language(CUDA)
project(CatmullClarkRewrite)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include_directories(..)
add_executable(gpu subd_rewrite.cpp ../CatmullClark.cu ../Mesh.cpp ../Utilities.h)
target_compile_features(gpu PRIVATE cxx_std_14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
set_target_properties(gpu PROPERTIES CUDA_ARCHITECTURES "80")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


configure_file(job-gpu job-gpu COPYONLY)