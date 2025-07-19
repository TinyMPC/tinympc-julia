using Test
include("../src/TinyMPC.jl")
using .TinyMPC
using LinearAlgebra

@testset "Code Generation Functionality" begin
    # Test system (cartpole)
    A = [1.0  0.01  0.0   0.0;
         0.0  1.0   0.039 0.0;
         0.0  0.0   1.002 0.01;
         0.0  0.0   0.458 1.002]
    B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
    Q = diagm([10.0, 1.0, 10.0, 1.0])
    R = diagm([1.0])
    N = 5  # Small horizon for fast testing
    rho = 1.0
    
    # Create temporary directory for testing
    test_dir = mktempdir()
    
    @testset "Basic Code Generation" begin
        solver = TinyMPCSolver()
        # Fix: correct dimensions for bounds - should be nu x (N-1) for control bounds
        u_min = fill(-0.5, 1, N-1)  
        u_max = fill(0.5, 1, N-1)
        
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, 
                       u_min=u_min, u_max=u_max, verbose=false)
        @test status == 0
        
        # Set references (required for code generation)
        set_x_ref(solver, zeros(4, N))
        set_u_ref(solver, zeros(1, N-1))
        
        # Generate code
        out_dir = joinpath(test_dir, "basic_codegen")
        status = codegen(solver, out_dir, verbose=false)
        @test status == 0
        
        # Check that essential files are created (only the files TinyMPC actually generates)
        essential_files = [
            joinpath("src", "tiny_main.cpp"),
            joinpath("src", "tiny_data.cpp"),
            joinpath("tinympc", "tiny_data.hpp")
        ]
        
        for file in essential_files
            file_path = joinpath(out_dir, file)
            @test isfile(file_path)
        end
    end
    
    @testset "Code Generation with Sensitivity" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Set references
        set_x_ref(solver, zeros(4, N))
        set_u_ref(solver, zeros(1, N-1))
        
        # Compute sensitivity matrices
        cache = compute_cache_terms(solver, A, B, Q, R; rho=rho)
        ε = 1e-6
        cache₊ = compute_cache_terms(solver, A, B, Q, R; rho=rho + ε)
        cache₋ = compute_cache_terms(solver, A, B, Q, R; rho=rho - ε)
        
        dK  = (cache₊.Kinf   - cache₋.Kinf)   / (2ε)
        dP  = (cache₊.Pinf   - cache₋.Pinf)   / (2ε)
        dC1 = (cache₊.Quu_inv - cache₋.Quu_inv) / (2ε)
        dC2 = (cache₊.AmBKt  - cache₋.AmBKt)  / (2ε)
        
        # Generate code with sensitivity
        out_dir = joinpath(test_dir, "sensitivity_codegen")
        status = codegen_with_sensitivity(solver, out_dir, dK, dP, dC1, dC2, verbose=false)
        @test status == 0
        
        # Check that essential files are created
        essential_files = [
            joinpath("src", "tiny_main.cpp"),
            joinpath("src", "tiny_data.cpp"),
            joinpath("tinympc", "tiny_data.hpp")
        ]
        
        for file in essential_files
            file_path = joinpath(out_dir, file)
            @test isfile(file_path)
        end
    end
    
    @testset "Code Generation Directory Creation" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Set references
        set_x_ref(solver, zeros(4, N))
        set_u_ref(solver, zeros(1, N-1))
        
        # Test with nested directory path - create parent dirs first
        out_dir = joinpath(test_dir, "nested", "path", "test")
        mkpath(dirname(out_dir))  # Ensure parent directories exist
        status = codegen(solver, out_dir, verbose=false)
        @test status == 0
        
        # Check that the directory structure was created
        @test isdir(out_dir)
        @test isfile(joinpath(out_dir, "src", "tiny_data.cpp"))
    end
    
    @testset "Code Generation File Content" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Set references
        set_x_ref(solver, zeros(4, N))
        set_u_ref(solver, zeros(1, N-1))
        
        out_dir = joinpath(test_dir, "content_test")
        status = codegen(solver, out_dir, verbose=false)
        @test status == 0
        
        # Check that generated files contain expected content
        data_cpp_path = joinpath(out_dir, "src", "tiny_data.cpp")
        data_content = read(data_cpp_path, String)
        @test contains(data_content, "#include")
        @test contains(data_content, "tinytype")
        
        main_cpp_path = joinpath(out_dir, "src", "tiny_main.cpp")
        main_content = read(main_cpp_path, String)
        @test contains(main_content, "#include")
        @test contains(main_content, "main")
        
        header_path = joinpath(out_dir, "tinympc", "tiny_data.hpp")
        header_content = read(header_path, String)
        @test contains(header_content, "#pragma once")
        @test contains(header_content, "extern")
    end
    
    @testset "Print Problem Data" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # This should not throw an error
        @test_nowarn print_problem_data(solver, verbose=true)
    end
    
    # Cleanup
    rm(test_dir, recursive=true)
end 