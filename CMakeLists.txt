cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(XPA LANGUAGES CXX)

# set the path for the created libraries and executable files
include(GNUInstallDirs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# use the new policy to allow CMake to link target libraries from other directories
cmake_policy(SET CMP0079 NEW)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-o2 -msse -msse2 -fopenmp")

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)

set(MKL_ROOT "/opt/intel/compilers_and_libraries_2019.4.243/linux/mkl")

# In the future modification, I will wirte my own find_package module.
# Now we set this path manually.
set(GSL_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/libgsl.a)
set(DL_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/libdl.a)

set(MKL_LIBRARY_FLAGS "-Wl,--start-group ${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a") 
set(MKL_LIBRARY_FLAGS "${MKL_LIBRARY_FLAGS} ${MKL_ROOT}/lib/intel64/libmkl_gnu_thread.a")
set(MKL_LIBRARY_FLAGS "${MKL_LIBRARY_FLAGS} ${MKL_ROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl")

find_package(Boost 1.58.0 COMPONENTS program_options REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(MPI REQUIRED)

add_subdirectory(src)

add_executable(MMNET ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

target_link_libraries(dataIO PUBLIC ${Boost_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries(Params PUBLIC dataIO)
target_link_libraries(GenoData PUBLIC GenotypeBasis)
target_link_libraries(AuxGenoData PUBLIC GenotypeBasis)
target_link_libraries(GeneticCorr PUBLIC NumericUtils GenoData AuxGenoData Eigen3::Eigen)
target_link_libraries(LMMCPU PUBLIC GenoData Eigen3::Eigen MPI::MPI_CXX)

list(APPEND LINK_STATIC_LIBRARIES dataIO Params Timer GenoData MemoryUtils
                                  LMMCPU NumericUtils GeneticCorr GenotypeBasis
                                  AuxGenoData)

target_link_libraries(XPA PUBLIC ${LINK_STATIC_LIBRARIES} 
                                   ${GSL_LIBRARY_PATH} 
                                   ${DL_LIBRARY_PATH}
                                   ${MKL_LIBRARY_FLAGS})
