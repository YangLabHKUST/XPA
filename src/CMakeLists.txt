add_library(dataIO STATIC IOUtils.h
                          IOUtils.cpp)
add_library(Timer STATIC Timer.h
                         Timer.cpp)
add_library(Params STATIC Parameters.h
                          Parameters.cpp)
add_library(MemoryUtils STATIC MemoryUtils.h
                               MemoryUtils.cpp)
add_library(LMMCPU STATIC LMMCPU.h
                          LMMCPU.cpp)
add_library(NumericUtils STATIC NumericUtils.h
                                NumericUtils.cpp)
add_library(GenoData STATIC GenoData.h
                                GenoData.cpp)
add_library(GeneticCorr STATIC GeneticCorr.h
								GeneticCorr.cpp)     
add_library(GenotypeBasis STATIC GenotypeBasis.h
                GenotypeBasis.cpp)
add_library(AuxGenoData STATIC AuxGenoData.h
                              AuxGenoData.cpp)   

MESSAGE(STATUS "Finish building libraries")          