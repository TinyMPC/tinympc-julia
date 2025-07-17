using Test
using TinyMPC
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
        u_min = reshape([-0.5], 1, 1)
        u_max = reshape([0.5], 1, 1)
        
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, 
                       u_min=u_min, u_max=u_max, verbose=false)
        @test status == 0
        
        # Set references (required for code generation)
        set_x_ref!(solver, zeros(4, N))
        set_u_ref!(solver, zeros(1, N-1))
        
        # Generate code
        out_dir = joinpath(test_dir, "basic_codegen")
        status = codegen(solver, out_dir, verbose=false)
        @test status == 0
        
        # Check that essential files are created
        essential_files = [
            "CMakeLists.txt",
            "setup.py",
            "bindings.cpp",
            joinpath("src", "tiny_main.cpp"),
            joinpath("src", "tiny_data.cpp"),
            joinpath("tinympc", "tiny_data.hpp")
        ]
        
        for file in essential_files
            file_path = joinpath(out_dir, file)
            @test isfile(file_path) "Missing file: $file"
        end
    end
    
    @testset "Code Generation with Sensitivity" begin
        solver = TinyMPCSolver()
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Set references
        set_x_ref!(solver, zeros(4, N))
        set_u_ref!(solver, zeros(1, N-1))
        
        # Compute sensitivity matrices
        sensitivities = compute_sensitivity_autograd(solver, A, B, Q, R, rho, verbose=false)
        
        # Generate code with sensitivity
        out_dir = joinpath(test_dir, "sensitivity_codegen")
        status = codegen_with_sensitivity(solver, out_dir, 
                                        sensitivities.dK, sensitivities.dP,
                                        sensitivities.dC1, sensitivities.dC2,
                                        verbose=false)
        @test status == 0
        
        # Check that files are created
        @test isfile(joinpath(out_dir, "CMakeLists.txt"))
        @test isfile(joinpath(out_dir, "setup.py"))
        @test isfile(joinpath(out_dir, "bindings.cpp"))
    end
    
    @testset "Code Generation Directory Creation" begin
        solver = TinyMPCSolver()
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Set references
        set_x_ref!(solver, zeros(4, N))
        set_u_ref!(solver, zeros(1, N-1))
        
        # Generate code to non-existent directory (should create it)
        out_dir = joinpath(test_dir, "new_dir", "nested", "path")
        @test !isdir(out_dir)
        
        status = codegen(solver, out_dir, verbose=false)
        @test status == 0
        @test isdir(out_dir)
        @test isfile(joinpath(out_dir, "CMakeLists.txt"))
    end
    
    @testset "Code Generation File Content" begin
        solver = TinyMPCSolver()
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Set references
        set_x_ref!(solver, zeros(4, N))
        set_u_ref!(solver, zeros(1, N-1))
        
        # Generate code
        out_dir = joinpath(test_dir, "content_check")
        status = codegen(solver, out_dir, verbose=false)
        @test status == 0
        
        # Check that CMakeLists.txt contains reasonable content
        cmake_content = read(joinpath(out_dir, "CMakeLists.txt"), String)
        @test contains(cmake_content, "cmake_minimum_required")
        @test contains(cmake_content, "project")
        
        # Check that setup.py contains reasonable content
        setup_content = read(joinpath(out_dir, "setup.py"), String)
        @test contains(setup_content, "pybind11")
        @test contains(setup_content, "Extension")
        
        # Check that bindings.cpp exists and has reasonable size
        bindings_path = joinpath(out_dir, "bindings.cpp")
        @test isfile(bindings_path)
        @test filesize(bindings_path) > 1000  # Should be a substantial file
    end
    
    @testset "Print Problem Data" begin
        solver = TinyMPCSolver()
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Set up problem
        set_x0!(solver, [0.1, 0.0, 0.0, 0.0])
        set_x_ref!(solver, zeros(4, N))
        set_u_ref!(solver, zeros(1, N-1))
        solve!(solver)
        
        # Print problem data (should not error)
        @test_nowarn print_problem_data(solver, verbose=true)
    end
    
    # Clean up
    @test_nowarn rm(test_dir, recursive=true)
end 