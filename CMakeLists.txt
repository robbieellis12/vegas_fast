cmake_minimum_required(VERSION 3.18)
project(vegasfast VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)
#set(GSL_INCLUDE_DIR /usr/local/include)
#set(GSL_LIBRARY /usr/local/lib/libgsl.so)
#set(GSL_CBLAS_LIBRARY /usr/local/lib/libgslcblas.so)
#find_package(GSL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

#if(ENABLE_CUDA_DEBUG)
#    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
#else()
#    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
#endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")


add_executable(vegasfast main.cu vegas.cu)

target_include_directories(vegasfast PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    /usr/local/include
)

target_compile_options(vegasfast PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
set_property(TARGET vegasfast PROPERTY CUDA_ARCHITECTURES 120)

target_compile_options(vegasfast PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --extended-lambda
                       --generate-line-info
                       -O2
                       -prec-div=false
                       -ftz=true
                       --restrict
                      --use_fast_math>)
target_compile_features(vegasfast PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(vegasfast PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(vegasfast PUBLIC
    CUDA::cudart CUDA::curand
)
set_property(TARGET vegasfast PROPERTY ENVIRONMENT "CUDA_MODULE_LOADING=EAGER")

if(OpenMP_C_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
endif()

#add_custom_target(run_vegas
#    COMMAND ${CMAKE_COMMAND} -E env CUDA_MODULE_LOADING=EAGER $<TARGET_FILE:vegas>
#    DEPENDS vegas
#    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#)


add_custom_command(TARGET vegasfast POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:vegasfast> vegasfast_re
    COMMAND ${CMAKE_COMMAND} -E echo "source nsys_easy -t cuda ./vegasfast_re" > vegasfast
    COMMAND chmod +x vegasfast
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)






