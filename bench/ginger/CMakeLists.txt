cmake_minimum_required(VERSION 3.24)
project(ginger_bench LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Ginger is meant to be consumed from source; pull in the submodule. Its own
# tests and examples are not needed here. jemalloc matches the paper's setup
# (see ginger/README.md); -march=native is Ginger's default.
set(GINGER_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(GINGER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GINGER_USE_JEMALLOC ON CACHE BOOL "" FORCE)
set(GINGER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../ginger)
add_subdirectory(${GINGER_SRC} ginger EXCLUDE_FROM_ALL)

# Ginger's default copts (-march=native, -funsigned-char, -fno-exceptions) and
# linkopts (threads, jemalloc) are plain variables set inside its directory
# scope, so they are invisible here and ginger_cc_binary would silently apply
# nothing -- which is how a first build of this driver came out without AVX2.
# Read them out of Ginger's scope instead. -funsigned-char in particular must
# match the library, since Ginger's headers inline code into this file.
get_directory_property(GINGER_DEFAULT_COPTS DIRECTORY ${GINGER_SRC} DEFINITION GINGER_DEFAULT_COPTS)
get_directory_property(GINGER_DEFAULT_LINKOPTS DIRECTORY ${GINGER_SRC} DEFINITION GINGER_DEFAULT_LINKOPTS)
if(NOT GINGER_DEFAULT_COPTS)
  message(FATAL_ERROR "could not read GINGER_DEFAULT_COPTS from ${GINGER_SRC}")
endif()
message(STATUS "ginger_bench copts: ${GINGER_DEFAULT_COPTS}")
# The linkopts name Threads::Threads, an imported target that is likewise only
# defined in Ginger's scope.
find_package(Threads REQUIRED)

ginger_cc_binary(
  NAME ginger_bench
  SRCS ginger_bench.cc
  # npy.h reports bad files by throwing; Ginger's default copts include
  # -fno-exceptions, and the later -fexceptions wins for this one file.
  COPTS -I${CMAKE_CURRENT_SOURCE_DIR}/../faiss -fexceptions
  DEPS
    ginger::ginger
    absl::status
    absl::statusor
    absl::span
    protobuf::libprotobuf
)
