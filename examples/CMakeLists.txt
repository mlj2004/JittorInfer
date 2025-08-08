# dependencies

find_package(Threads REQUIRED)

# third-party

# Add yaml-cpp
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/yaml-cpp ${CMAKE_BINARY_DIR}/yaml-cpp)

# flags

llama_add_compile_flags()

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/common)
include_directories(${CMAKE_SOURCE_DIR}/third_party)
if (EMSCRIPTEN)
else()
    add_subdirectory(server)
    add_subdirectory(mpi_server)
endif()
