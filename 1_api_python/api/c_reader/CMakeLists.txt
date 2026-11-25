cmake_minimum_required(VERSION 3.13)
project(daq_reader C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

add_library(daq_reader SHARED daq_reader.c)
target_include_directories(daq_reader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(daq_reader PRIVATE Threads::Threads)
target_compile_options(daq_reader PRIVATE -Wall -Wextra -Wpedantic)

add_executable(stream_test stream_test.c)
