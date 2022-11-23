cmake_minimum_required(VERSION 3.16)

project(pro4)

add_definitions(-mavx2 -DWITH_AVX2)

set(CMAKE_CXX_STANDARD 11)


include_directories(/usr/include)
# add_library(usr/lib)

link_libraries(-lopenblas -lpthread -lgfortran)

add_executable(mul source.c main.c )#加入可执行文件，空格分开   

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("OpenMP found.")
    target_link_libraries(mul PUBLIC OpenMP::OpenMP_CXX)
endif()
#cc -o test_cblas_open test_cblas_dgemm.c -I /usr/include/ -L/usr/lib -lopenblas -lpthread -lgfortran