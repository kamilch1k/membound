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
# -lineinfo: source-level correlation in Nsight Compute; warnings are errors
if(MSVC)
  set(HOST_WARN -Xcompiler=/W4 -Xcompiler=/WX)
else()
  set(HOST_WARN -Xcompiler=-Wall,-Wextra,-Werror)
endif()
foreach(t membound membound-softmax membound-dequant)
  target_compile_options(${t} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr ${HOST_WARN}>
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror all-warnings>")
endforeach()

# GPU-free smoke tests: --profile runs every kernel once, correctness-gated
enable_testing()
add_test(NAME gemv_correct    COMMAND membound --profile)
add_test(NAME softmax_correct COMMAND membound-softmax --profile)
add_test(NAME dequant_correct COMMAND membound-dequant --profile)
