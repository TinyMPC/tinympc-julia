module TinyMPC

# Temporary package
# There can be better ways to do this

# Compile the generated code
function compile_lib(dir::String)
    print("Compiling library to ", dir, "\n")
    build_dir = joinpath(dir, "build")
    if !isdir(build_dir)
        mkdir(build_dir)
    end
    run(`cmake -S $dir -B $build_dir`)
    run(`cmake --build $build_dir`)
    return true
end

end # module
