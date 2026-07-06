cmake_minimum_required(VERSION 3.24)
project(membound LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89) # RTX 4080 Laptop (Ada)
endif()

add_executable(membound src/main.cu)
add_executable(membound-softmax src/softmax.cu)
add_executable(membound-dequant src/dequant.cu)
# -lineinfo: source-level correlation in Nsight Compute
foreach(t membound membound-softmax membound-dequant)
  target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr>)
endforeach()
