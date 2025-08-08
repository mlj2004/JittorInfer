set(TARGET llama-server)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR}/common ${CMAKE_SOURCE_DIR}/third_party ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp/include)

set(TARGET_SRCS
    server.cpp
    ${CMAKE_SOURCE_DIR}/common/utils.hpp
    ${CMAKE_SOURCE_DIR}/common/common_local.cpp
    ${CMAKE_SOURCE_DIR}/common/json-schema-to-grammar_local.cpp
    ${CMAKE_SOURCE_DIR}/common/sampling_local.cpp
    ${CMAKE_SOURCE_DIR}/common/chat_local.cpp
    ${CMAKE_SOURCE_DIR}/common/chat-template.hpp
    ${CMAKE_SOURCE_DIR}/third_party/json.hpp
    ${CMAKE_SOURCE_DIR}/third_party/httplib.h
    ${CMAKE_SOURCE_DIR}/third_party/minja.hpp
)

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

# target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})


target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT} yaml-cpp)
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/common ${CMAKE_SOURCE_DIR}/third_party)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
