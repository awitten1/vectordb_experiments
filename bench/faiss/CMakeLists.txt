cmake_minimum_required(VERSION 3.24)
project(faiss_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ../../faiss/install, populated by build_faiss.sh. Passed as CMAKE_PREFIX_PATH
# from that script; the default here is only for a manual cmake invocation.
find_package(faiss REQUIRED
  HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../../faiss/install")
find_package(OpenMP REQUIRED)

# FAISS's C++ libraries do NOT dispatch on CPU features at run time -- only the
# Python bindings pick a variant. Linking plain `faiss` would silently give the
# generic build, which is an unfair baseline against pgvector and
# pg_vectorsearch, both built -march=native. So link the best variant the
# install actually has. faiss_bench --version prints the result; it must not
# say plain OPTIMIZE.
set(FAISS_TARGET faiss)
foreach(variant faiss_avx512 faiss_avx2)
  if(TARGET ${variant})
    set(FAISS_TARGET ${variant})
    break()
  endif()
endforeach()
message(STATUS "linking against ${FAISS_TARGET}")

add_executable(faiss_bench faiss_bench.cpp)
target_link_libraries(faiss_bench PRIVATE ${FAISS_TARGET} OpenMP::OpenMP_CXX)

# Pin the include path to the submodule's own install, BEFORE the defaults.
# There is a stale FAISS at /usr/local/include/faiss on this machine with an
# older index_factory signature; without this we compile against those headers
# and link against the library we just built, which fails at link time if we
# are lucky and misbehaves if we are not.
get_target_property(_faiss_inc ${FAISS_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
if(NOT _faiss_inc)
  set(_faiss_inc "${CMAKE_CURRENT_SOURCE_DIR}/../../faiss/install/include")
endif()
target_include_directories(faiss_bench BEFORE PRIVATE ${_faiss_inc})
