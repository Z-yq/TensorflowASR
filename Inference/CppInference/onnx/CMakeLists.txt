project(ASR_DEMO)
cmake_minimum_required(VERSION 3.10) 
include_directories(${CMAKE_SOURCE_DIR}/ext/)
include_directories(${CMAKE_SOURCE_DIR}/ext/onnxruntime/include)
include_directories(${CMAKE_SOURCE_DIR}/src/core)
link_libraries(${CMAKE_SOURCE_DIR}/ext/onnxruntime/lib/libonnxruntime.so.1.10.0)

add_executable(test.out CppInference.cpp ${CMAKE_SOURCE_DIR}/src/core/asr_session.cpp)
