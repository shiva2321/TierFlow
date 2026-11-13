cmake_minimum_required(VERSION 3.20)
project(tierflow_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /EHsc /permissive-)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(TF_SOURCES
    tf_governor.cpp
    tf_cache.cpp
    tf_kv.cpp
    tf_manifest.cpp
    tf_probe.cpp
    tf_planner.cpp
    tf_spec.cpp
)

if(WIN32)
    list(APPEND TF_SOURCES
        tf_io_win.cpp
        tf_io_ioring_win.cpp
        tf_mem_win.cpp
    )
elseif(UNIX)
    list(APPEND TF_SOURCES
        tf_io_linux.cpp
    )
endif()

add_library(tierflow_core STATIC ${TF_SOURCES})
target_include_directories(tierflow_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
    target_link_libraries(tierflow_core PUBLIC kernel32 psapi pdh)
elseif(UNIX)
    find_library(URING_LIB uring)
    if(URING_LIB)
        target_compile_definitions(tierflow_core PUBLIC TF_HAS_LIBURING=1)
        target_link_libraries(tierflow_core PUBLIC ${URING_LIB})
    endif()
    target_link_libraries(tierflow_core PUBLIC pthread)
endif()

add_executable(tf_tests tests/test_main.cpp)
target_link_libraries(tf_tests PRIVATE tierflow_core)

add_executable(tf_io_bench tf_io_bench.cpp)
target_link_libraries(tf_io_bench PRIVATE tierflow_core)

# Link TierFlow live llama.cpp streaming runner if third_party/llama.cpp/build exists
set(LLAMA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp")
set(LLAMA_BLD_DIR "${LLAMA_SRC_DIR}/build")
if(EXISTS "${LLAMA_BLD_DIR}/src/llama.lib")
    add_executable(tf_runner tf_llama_stream.cpp tf_runner.cpp)
    target_include_directories(tf_runner PRIVATE
        "${LLAMA_SRC_DIR}/include"
        "${LLAMA_SRC_DIR}/ggml/include"
    )
    target_link_libraries(tf_runner PRIVATE
        tierflow_core
        "${LLAMA_BLD_DIR}/src/llama.lib"
        "${LLAMA_BLD_DIR}/ggml/src/ggml.lib"
        "${LLAMA_BLD_DIR}/ggml/src/ggml-cpu.lib"
        "${LLAMA_BLD_DIR}/ggml/src/ggml-base.lib"
    )
    if(WIN32)
        target_link_libraries(tf_runner PRIVATE user32 advapi32)
    endif()
endif()

enable_testing()
add_test(NAME tierflow_unit_tests COMMAND tf_tests)

