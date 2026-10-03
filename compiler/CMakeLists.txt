# WinHint compiler plugins (out-of-tree LLVM new-pass-manager plugins).
#
# Host build from the `winhint` micromamba env (see compiler/build.sh and
# build/ENV.md); never inside Docker:
#   cmake -G Ninja -S compiler -B build/compiler -DLLVM_DIR=$(llvm-config --cmakedir)
#   ninja -C build/compiler -j1
# Produces build/compiler/WinHint.so and (when compiler/baselines/jones_iq is
# present) build/compiler/JonesIQ.so.
#
# Targets
#   WinHintCommon  static lib: common/TargetModel + common/HintEmitter. Shared
#                  with the baselines (owned by compiler/baselines/*); its API
#                  is documented in the two headers and kept stable.
#   WinHint        compiler/winhint/*  (MODULE, dlopen()ed by clang/opt)
#   JonesIQ        compiler/baselines/jones_iq (B6). If that directory has its
#                  own CMakeLists.txt it is add_subdirectory()'d (it can use
#                  the WinHintCommon target and the winhint_add_plugin()
#                  helper); otherwise JonesIQ.cpp is built here.
cmake_minimum_required(VERSION 3.20)
project(WinHintCompiler CXX)

find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")

# The pinned major version comes from tooling/versions.lock (LLVM=<x.y.z>).
set(WINHINT_LLVM_MAJOR "" CACHE STRING "Required LLVM major version (default: tooling/versions.lock)")
if(NOT WINHINT_LLVM_MAJOR)
  set(_lock "${CMAKE_CURRENT_SOURCE_DIR}/../tooling/versions.lock")
  if(EXISTS "${_lock}")
    file(STRINGS "${_lock}" _llvm_line REGEX "^LLVM=")
    if(_llvm_line)
      string(REGEX REPLACE "^LLVM=([0-9]+).*" "\\1" WINHINT_LLVM_MAJOR "${_llvm_line}")
    endif()
  endif()
endif()
if(WINHINT_LLVM_MAJOR AND NOT LLVM_VERSION_MAJOR EQUAL WINHINT_LLVM_MAJOR)
  message(FATAL_ERROR "WinHint expects LLVM ${WINHINT_LLVM_MAJOR} (tooling/versions.lock), "
                      "found ${LLVM_PACKAGE_VERSION}")
endif()
if(LLVM_VERSION_MAJOR LESS 21)
  message(FATAL_ERROR "WinHint needs LLVM >= 21 (found ${LLVM_PACKAGE_VERSION})")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFS})
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()
add_compile_options(-Wall -Wno-unused-parameter -fvisibility-inlines-hidden)

# Plugins are dlopen()ed by clang/opt, which already export the LLVM symbols.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_SHARED_MODULE_PREFIX "")

add_library(WinHintCommon STATIC
  common/TargetModel.cpp
  common/HintEmitter.cpp)
target_include_directories(WinHintCommon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

# winhint_add_plugin(<name> <sources...>): a pass-plugin MODULE linked with
# WinHintCommon, written to build/compiler/<name>.so.
function(winhint_add_plugin name)
  add_library(${name} MODULE ${ARGN})
  target_link_libraries(${name} PRIVATE WinHintCommon)
  set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

winhint_add_plugin(WinHint
  winhint/Options.cpp
  winhint/WindowDemandAnalysis.cpp
  winhint/HintPlacement.cpp
  winhint/Plugin.cpp)
target_include_directories(WinHint PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/winhint)

# B6 (owned by compiler/baselines/jones_iq).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/baselines/jones_iq/CMakeLists.txt)
  add_subdirectory(baselines/jones_iq)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/baselines/jones_iq/JonesIQ.cpp)
  winhint_add_plugin(JonesIQ baselines/jones_iq/JonesIQ.cpp)
endif()
