cmake_minimum_required(VERSION 3.24)
project(matrix_multiplication C)

set(CMAKE_C_STANDARD 99)
include_directories(/usr/include/x86_64-linux-gnu/mpich)
set(CMAKE_C_COMPILER "/usr/bin/mpicc")
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")

add_executable(matrix_multiplication main.c)
target_link_libraries(matrix_multiplication m)