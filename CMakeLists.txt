cmake_minimum_required(VERSION 2.8)
project(gasdynamics)

option(USE_OPENMP "Enable OpenMP" ON)
option(USE_MPI "Enable MPI" ON)

# Set language property 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Valid options are Release or Debug" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug")
endif()

if (NOT (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "Debug"))
  message(FATAL_ERROR "Build type \"${CMAKE_BUILD_TYPE}\" is not supported.")
endif()

message("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
message("CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    list(APPEND ALL_THIRD_PARTY_LIBRARIES PUBLIC OpenMP::OpenMP_CXX)
else()
    if( (CMAKE_CXX_COMPILER_ID MATCHES "[cC][lL][aA][nN][gG]")
            OR (CMAKE_CXX_COMPILER_ID MATCHES "[gG][nN][uU]"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    endif()
endif(USE_OPENMP)


if(USE_MPI)

  find_package(MPI REQUIRED)

  message("Found MPI, with mpiexec=${MPIEXEC}")
  include_directories(${MPI_CXX_INCLUDE_PATH})
  set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS} ")
  set(CMAKE_EXE_LINKER_FLAGS " ${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS} ")
  list(APPEND ALL_THIRD_PARTY_LIBRARIES ${MPI_CXX_LIBRARIES})

  message("Adding MPI include:${MPI_CXX_INCLUDE_PATH}.")
  message("Adding MPI libs:${MPI_CXX_LIBRARIES}.")

endif()



set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
add_subdirectory(Code)