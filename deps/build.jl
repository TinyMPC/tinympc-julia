using Libdl

# This script builds the TinyMPC C++ library for Julia
println("Building TinyMPC Julia wrapper...")

# Get the Julia package root directory (parent of deps)
const package_root = dirname(@__DIR__)
const build_dir = joinpath(package_root, "build")
const lib_dir = joinpath(package_root, "lib")
const lib_name = "libtinympc_jl"
const lib_filename = lib_name * "." * Libdl.dlext
const build_lib_filename = "tinympc_jl" * "." * Libdl.dlext  # CMake creates without lib prefix

println("Package root: $package_root")
println("Build directory: $build_dir")
println("Lib directory: $lib_dir")

# Function to run a command and show output
function run_cmd(cmd, workdir=pwd())
    println("Running in $workdir: `$cmd`")
    cd(workdir) do
        run(cmd)
    end
end

try
    # 1. Create build directory if it doesn't exist
    if !isdir(build_dir)
        println("Creating build directory...")
        mkdir(build_dir)
    end

    # 2. Run CMake and Make from the build directory
    println("Configuring with CMake...")
    run_cmd(`cmake ..`, build_dir)
    
    println("Building with make...")
    run_cmd(`make -j$(Sys.CPU_THREADS)`, build_dir)
    
    # 3. Create lib directory if it doesn't exist
    if !isdir(lib_dir)
        println("Creating lib directory...")
        mkdir(lib_dir)
    end

    # 4. Copy the built library to the lib directory
    src_path = joinpath(build_dir, build_lib_filename)
    dest_path = joinpath(lib_dir, lib_filename)
    
    if isfile(src_path)
        println("Copying library from $src_path to $dest_path")
        cp(src_path, dest_path, force=true)
        println("TinyMPC Julia wrapper built successfully!")
    else
        error("Built library not found at $src_path")
    end
    
catch e
    @error "Failed to build the TinyMPC Julia wrapper" exception=(e, catch_backtrace())
    println("Make sure you have:")
    println("1. CMake installed")
    println("2. A C++ compiler (g++/clang++)")
    println("3. Julia development headers")
    println("4. Git submodules initialized (git submodule update --init --recursive)")
    rethrow(e)
end 