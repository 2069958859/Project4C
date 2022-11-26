cmake_minimum_required(VERSION 3.16)

project(pro4)

add_definitions(-mavx2 -DWITH_AVX2)

add_definitions(-fopenmp -lgomp)

set(CMAKE_CXX_STANDARD 11)
# add_compile_options(-mavx2 -fopenmp)
# add_compile_options(-O3)


# set(CMAKE_C_FLAGS_Release "${CMAKE_C_FLAGS_Release} -O3")

#include_directories(/usr/include)
# add_library(usr/lib)
add_executable(mul source.c main.c )#加入可执行文件，空格分开   


target_link_libraries(mul /usr/local/OpenBLAS/lib/libopenblas.a)
target_link_libraries(mul pthread)

#link_libraries(-lopenblas -lpthread -lgfortran)


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("OpenMP found.")
   target_link_libraries(mul  OpenMP::OpenMP_CXX)
endif()
#cc -o test_cblas_open test_cblas_dgemm.c -I /usr/include/ -L/usr/lib -lopenblas -lpthread -lgfortran