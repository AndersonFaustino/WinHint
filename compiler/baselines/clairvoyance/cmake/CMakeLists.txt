# Wrapper around third_party/clairvoyance/compiler/CMakeLists.txt (the artifact's
# top level is not usable with CMake >= 3.27: its add_cxx_flag() macro relies on the
# OLD behaviour of policy CMP0054, which modern CMake no longer offers). Same
# settings, then the artifact's own projects/ tree, unmodified.
cmake_minimum_required(VERSION 3.20)
project(ClairvoyancePasses C CXX)

set(CV_SRC "" CACHE PATH "third_party/clairvoyance/compiler")
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_DIR})")
if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL "3.8.1")
  message(WARNING "The Clairvoyance artifact targets LLVM 3.8 (found ${LLVM_PACKAGE_VERSION})")
endif()
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

# The artifact's projects/ CMake files cannot be add_subdirectory()'d either
# (LoopExtract reads the LOCATION target property: policy CMP0026, removed in
# CMake 4). The targets below are a 1:1 transcription of
# projects/src/**/CMakeLists.txt (same sources, same library kinds); bare
# directory entries in the artifact's source lists are dropped.
set(P ${CV_SRC}/projects)
set(S ${P}/src)
include_directories(${P}/include)

add_library(UtilLoops MODULE ${S}/Util/Loops/ForcedLoopUnroll.cpp)
add_library(BranchAnnotate MODULE
  ${S}/Util/Annotation/BranchAnnotate/SBPAnnotator.cpp
  ${S}/Util/Annotation/MetadataInfo.cpp)
add_library(CFGIndirectionCount MODULE
  ${S}/Util/Annotation/CFGIndirectionCount/CFGIndirectionCount.cpp
  ${S}/Util/Annotation/MetadataInfo.cpp
  ${S}/Util/Analysis/LoopDependency.cpp
  ${S}/Util/Analysis/AliasUtils.cpp)
add_library(MarkLoopsToSwoopify SHARED
  ${S}/SWOOP/Utils/MarkLoopsToSwoopify/MarkLoopsToSwoopify.cpp
  ${S}/Util/Annotation/MetadataInfo.cpp)
set(SWOOP_COMMON
  ${S}/Util/Annotation/MetadataInfo.cpp
  ${S}/Util/Analysis/AliasUtils.cpp
  ${S}/Util/DAE/DAEUtils.cpp
  ${S}/Util/Analysis/LoopDependency.cpp
  ${S}/Util/Transform/BranchMerge/BranchMerge.cpp
  ${S}/SWOOP/Transform/PhaseStitching.cpp
  ${S}/SWOOP/Transform/SwoopDAE/LCDHandler.cpp
  ${S}/SWOOP/Transform/SwoopDAE/FindInstructions.cpp)
add_library(SwoopDAE SHARED ${S}/SWOOP/Transform/SwoopDAE/SwoopDAE.cpp ${SWOOP_COMMON})
add_library(OptimisticSwoop SHARED
  ${S}/SWOOP/Transform/SwoopDAE/SwoopDAE.cpp
  ${S}/SWOOP/Transform/OptimisticSwoop/OptimisticSwoop.cpp
  ${SWOOP_COMMON})
add_dependencies(OptimisticSwoop SwoopDAE)
add_library(LoopExtract MODULE ${S}/DAE/Utils/LoopExtract/LoopExtract.cpp)
foreach(t UtilLoops BranchAnnotate CFGIndirectionCount MarkLoopsToSwoopify SwoopDAE
          OptimisticSwoop LoopExtract)
  target_include_directories(${t} PRIVATE ${S}/SWOOP/Transform ${S}/SWOOP/Transform/SwoopDAE)
endforeach()
