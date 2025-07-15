using Test

examples_dir = normpath(@__DIR__, "../examples")
example_files = filter(f -> endswith(f, ".jl"), readdir(examples_dir; join=true))

@testset "Example scripts" begin
    for ex in example_files
        @info "Running example: $(basename(ex))"
        try
            include(ex)
            @test true
        catch e
            @error "Example $(ex) failed" exception=(e, catch_backtrace())
            @test false  # Fail the test
        end

        # If the example produced an "out" directory with a CMakeLists.txt, try to compile it
        out_dir = joinpath(dirname(ex), "out")
        if isfile(joinpath(out_dir, "CMakeLists.txt"))
            @info "Compiling generated code in $(out_dir)"
            try
                cd(out_dir) do
                    run(`cmake . -B build -DCMAKE_BUILD_TYPE=Release`)
                    run(`cmake --build build -j2`)
                end
                @test true
            catch e
                @error "Compilation failed for $(out_dir)" exception=(e, catch_backtrace())
                @test false
            end
        end
    end
end 