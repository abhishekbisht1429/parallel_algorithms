cmake_minimum_required(VERSION 3.21)
project(gaussian_elimination)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)

add_executable(gaussian_elimination
        gaussian_elimination.cpp)

target_link_libraries(gaussian_elimination PUBLIC MPI::MPI_CXX)