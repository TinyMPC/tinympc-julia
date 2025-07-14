cmake_minimum_required(VERSION 3.15...3.26)

project(tinympc_julia_wrapper)

# --- TinyMPC -------------------------------------------------
add_subdirectory(tinympc/TinyMPC tinympc_build)

find_package(Julia REQUIRED)
find_package(CxxWrap REQUIRED)
find_package(Eigen3 REQUIRED)

set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp
)

add_library(tinympc_julia MODULE ${SRC})

target_include_directories(tinympc_julia PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tinympc/TinyMPC/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tinympc/TinyMPC/src
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(tinympc_julia PRIVATE tinympcstatic CxxWrap::cxxwrap_julia Eigen3::Eigen)

set_target_properties(tinympc_julia PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
    PREFIX ""
)

if(APPLE)
    set_target_properties(tinympc_julia PROPERTIES
        INSTALL_NAME_DIR "@rpath"
    )
endif()

message(STATUS "Build the wrapper with: mkdir build && cd build && cmake .. && make")
