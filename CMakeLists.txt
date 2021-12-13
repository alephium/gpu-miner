cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(gpu-miner LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE TRUE)

# Add miner version at compile time, priority is tag, short hash and finally unknown
find_package(Git)
if(Git_FOUND)
    # Find current tag name :
    execute_process(COMMAND
            "${GIT_EXECUTABLE}" describe --tags --abbrev=0 --exact-match
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE miner-version
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    # If no tag was found, set to hash abbreviation
    if(NOT miner-version)
        execute_process(COMMAND
            "${GIT_EXECUTABLE}" describe --match=NeVeRmAtCh --always --abbrev=7
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE miner-version
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(miner-version canary-${miner-version})
        else()
        set(miner-version release-${miner-version})
    endif()
else()
    set(miner-version unknown)
endif()

message(miner version : ${miner-version})

#Setup conan
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
            "${CMAKE_BINARY_DIR}/conan.cmake")
endif()
include(${CMAKE_BINARY_DIR}/conan.cmake)
conan_cmake_run(REQUIRES libuv/1.42.0
        BASIC_SETUP)

# Target definitions
add_executable(gpu-miner src/main.cu)
set_property(TARGET gpu-miner PROPERTY CXX_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
target_compile_definitions(gpu-miner PUBLIC MINER_VERSION="${miner-version}")
target_link_libraries(gpu-miner PRIVATE ${CONAN_LIBS})
#Copy binary to bin directory
add_custom_command(TARGET gpu-miner
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gpu-miner> ${CMAKE_SOURCE_DIR}/bin)

add_executable(blake3-test src/blake3.cu src/blake3/blake3-common.hpp)
set_property(TARGET blake3-test PROPERTY CXX_STANDARD 11)
target_compile_definitions(blake3-test PUBLIC BLAKE3_TEST)
target_link_libraries(blake3-test PRIVATE ${CONAN_LIBS})
#Copy binary to bin directory
add_custom_command(TARGET blake3-test
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:blake3-test> ${CMAKE_SOURCE_DIR}/bin)
