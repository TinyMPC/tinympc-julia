# FindJulia.cmake
# Finds Julia installation and sets up variables for building Julia packages
#
# Variables set:
#   JULIA_FOUND           - True if Julia was found
#   JULIA_EXECUTABLE      - Path to julia executable
#   JULIA_INCLUDE_DIR     - Julia include directory
#   JULIA_LIBRARIES       - Julia libraries to link against
#   JULIA_VERSION         - Julia version string

# Find julia executable
find_program(JULIA_EXECUTABLE NAMES julia
    HINTS ENV JULIA_DIR
    PATHS
    /usr/bin
    /usr/local/bin
    $ENV{HOME}/.juliaup/bin
    $ENV{HOME}/julia/bin
    DOC "Julia executable"
)

if(JULIA_EXECUTABLE)
    # Get Julia version
    execute_process(
        COMMAND ${JULIA_EXECUTABLE} --version
        OUTPUT_VARIABLE JULIA_VERSION_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(JULIA_VERSION_STRING MATCHES "julia version ([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(JULIA_VERSION ${CMAKE_MATCH_1})
    endif()
    
    # Get Julia include directory
    execute_process(
        COMMAND ${JULIA_EXECUTABLE} -e "using Libdl; print(joinpath(Sys.BINDIR, \"..\", \"include\", \"julia\"))"
        OUTPUT_VARIABLE JULIA_INCLUDE_DIR_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Resolve the path
    get_filename_component(JULIA_INCLUDE_DIR ${JULIA_INCLUDE_DIR_RAW} ABSOLUTE)
    
    # Get Julia library directory and name
    execute_process(
        COMMAND ${JULIA_EXECUTABLE} -e "using Libdl; print(joinpath(Sys.BINDIR, \"..\", \"lib\"))"
        OUTPUT_VARIABLE JULIA_LIBRARY_DIR_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    get_filename_component(JULIA_LIBRARY_DIR ${JULIA_LIBRARY_DIR_RAW} ABSOLUTE)
    
    # Find the Julia library
    find_library(JULIA_LIBRARY
        NAMES julia libjulia
        PATHS ${JULIA_LIBRARY_DIR}
        NO_DEFAULT_PATH
    )
    
    if(JULIA_LIBRARY)
        set(JULIA_LIBRARIES ${JULIA_LIBRARY})
    else()
        # On some systems, we might need to link against libjulia dynamically
        execute_process(
            COMMAND ${JULIA_EXECUTABLE} -e "using Libdl; print(dlpath(\"libjulia\"))"
            OUTPUT_VARIABLE JULIA_LIBRARY_DYNAMIC
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(JULIA_LIBRARY_DYNAMIC)
            set(JULIA_LIBRARIES ${JULIA_LIBRARY_DYNAMIC})
        endif()
    endif()
    
    # Verify include directory exists
    if(EXISTS ${JULIA_INCLUDE_DIR}/julia.h)
        set(JULIA_INCLUDE_DIRS ${JULIA_INCLUDE_DIR})
    else()
        message(WARNING "Julia include directory not found or julia.h missing")
    endif()
endif()

# Handle the standard REQUIRED and QUIET arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Julia
    REQUIRED_VARS JULIA_EXECUTABLE JULIA_INCLUDE_DIR
    VERSION_VAR JULIA_VERSION
)

if(JULIA_FOUND)
    message(STATUS "Found Julia: ${JULIA_EXECUTABLE} (version ${JULIA_VERSION})")
    message(STATUS "Julia include dir: ${JULIA_INCLUDE_DIR}")
    message(STATUS "Julia libraries: ${JULIA_LIBRARIES}")
endif()

mark_as_advanced(
    JULIA_EXECUTABLE
    JULIA_INCLUDE_DIR
    JULIA_LIBRARIES
    JULIA_LIBRARY_DIR
) 