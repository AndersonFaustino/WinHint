# B6 JonesIQ plugin (JonesIQ.so).
#
# Normally built from compiler/CMakeLists.txt via add_subdirectory(baselines/jones_iq),
# which provides the WinHintCommon target (compiler/common). It can also be built
# standalone (cmake -S compiler/baselines/jones_iq -B $WINHINT_BUILD/compiler-jones),
# in which case compiler/common is compiled here.
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(JonesIQ CXX)
  find_package(LLVM REQUIRED CONFIG)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  separate_arguments(LLVM_DEFS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
  add_definitions(${LLVM_DEFS})
  if(NOT LLVM_ENABLE_RTTI)
    add_compile_options(-fno-rtti)
  endif()
endif()

if(NOT TARGET WinHintCommon)
  set(_common ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
  add_library(WinHintCommon STATIC ${_common}/TargetModel.cpp ${_common}/HintEmitter.cpp)
  target_include_directories(WinHintCommon PUBLIC ${_common})
  set_target_properties(WinHintCommon PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(JonesIQ MODULE JonesIQ.cpp)
target_link_libraries(JonesIQ PRIVATE WinHintCommon)
# dlopen()ed by clang/opt, which export the LLVM symbols: no LLVM libs linked.
set_target_properties(JonesIQ PROPERTIES PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
