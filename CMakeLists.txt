cmake_minimum_required(VERSION 3.10)
project(tinympc_julia CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add cmake modules path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Find Julia package
find_package(Julia REQUIRED)

# Check if TinyMPC submodule exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tinympc/TinyMPC/CMakeLists.txt")
    message(FATAL_ERROR "TinyMPC submodule not found. Please clone with --recurse-submodules or run 'git submodule update --init --recursive'")
endif()

# Build TinyMPC static library
add_subdirectory(tinympc/TinyMPC)

# Print Julia information for debugging
message(STATUS "Julia executable: ${JULIA_EXECUTABLE}")
message(STATUS "Julia include dir: ${JULIA_INCLUDE_DIR}")
message(STATUS "Julia libraries: ${JULIA_LIBRARIES}")

# Add the Julia bindings using direct C API (no CxxWrap)
add_library(tinympc_jl SHARED src/bindings.cpp)

target_include_directories(tinympc_jl PRIVATE
    ${JULIA_INCLUDE_DIR}
    "${CMAKE_CURRENT_SOURCE_DIR}/tinympc/TinyMPC/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/tinympc/TinyMPC/src"
)

target_link_libraries(tinympc_jl PRIVATE
    ${JULIA_LIBRARIES}
    tinympcstatic
)

# Set proper library properties
set_target_properties(tinympc_jl PROPERTIES
    PREFIX ""
    SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX}
)

# Install the library
install(TARGETS tinympc_jl
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Compiler Flags (Linux/Mac only)
# GCC/Clang specific settings
add_compile_options(-Wall -Wextra)
# Remove -Wpedantic for now to avoid strict warnings

# Print success message
message(STATUS "TinyMPC Julia wrapper configured successfully")
message(STATUS "  - Using direct Julia C API (no CxxWrap)")
message(STATUS "  - Target: libtinympc_jl${CMAKE_SHARED_LIBRARY_SUFFIX}")
message(STATUS "  - Build with: make -j\${nproc}")
