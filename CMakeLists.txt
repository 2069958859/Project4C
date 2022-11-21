cmake_minimum_required(VERSION 3.12)

#add_definitions(-DWITH_NEON)

add_definitions(-DWITH_AVX2)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-mavx2 -DWITH_AVX2)

project(pro4)
add_executable(mul source.c main.c )#加入可执行文件，空格分开   

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("OpenMP found.")
    target_link_libraries(mul PUBLIC OpenMP::OpenMP_CXX)
endif()
