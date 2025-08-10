using Test
include("../src/TinyMPC.jl")
using .TinyMPC
using LinearAlgebra

@testset "Settings Functionality" begin
    # Test system (cartpole)
    A = [1.0  0.01  0.0   0.0;
         0.0  1.0   0.039 0.0;
         0.0  0.0   1.002 0.01;
         0.0  0.0   0.458 1.002]
    B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
    Q = diagm([10.0, 1.0, 10.0, 1.0])
    R = diagm([1.0])
    N = 2  # Small horizon for fast testing
    
    @testset "Basic Settings Update" begin
        solver = TinyMPCSolver()
        
        # Setup with custom tolerance
        status = setup(solver, A, B, zeros(4), Q, R, 1.0, 4, 1, N, 
                       abs_pri_tol=5.0, verbose=false)
        @test status == 0
        
        # Test that solve still works with custom settings
        set_x0(solver, [0.1, 0.0, 0.0, 0.0])
        set_x_ref(solver, zeros(4, N))
        set_u_ref(solver, zeros(1, N-1))
        
        status = solve(solver)
        @test status == 0
    end
    
    @testset "All Settings Parameters" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, 1.0, 4, 1, N, verbose=false)
        @test status == 0
        
        # Update all settings
        @test_nowarn update_settings(solver,
            abs_pri_tol=1e-4,
            abs_dua_tol=1e-4,
            max_iter=50,
            check_termination=1,
            en_state_bound=1,
            en_input_bound=1,
            en_state_soc=0,
            en_input_soc=0,
            en_state_linear=0,
            en_input_linear=0,
            adaptive_rho=1,
            adaptive_rho_min=0.1,
            adaptive_rho_max=10.0,
            adaptive_rho_enable_clipping=1,
            verbose=false)
        
        # Test that solver still works after settings update
        set_x0(solver, [0.1, 0.0, 0.0, 0.0])
        set_x_ref(solver, zeros(4, N))
        set_u_ref(solver, zeros(1, N-1))
        
        status = solve(solver)
        @test status == 0
    end
    
    @testset "Adaptive Rho Settings" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, 1.0, 4, 1, N, 
                       adaptive_rho=true, adaptive_rho_min=0.5, adaptive_rho_max=5.0,
                       verbose=false)
        @test status == 0
        
        # Test that solver works with adaptive rho
        set_x0(solver, [0.1, 0.0, 0.0, 0.0])
        set_x_ref(solver, zeros(4, N))
        set_u_ref(solver, zeros(1, N-1))
        
        status = solve(solver)
        @test status == 0
    end
    
    @testset "Max Iterations Setting" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, 1.0, 4, 1, N, 
                       max_iter=1, verbose=false)  # Very low max_iter
        @test status == 0
        
        # Even with low max_iter, should complete without error
        set_x0(solver, [0.1, 0.0, 0.0, 0.0])
        set_x_ref(solver, zeros(4, N))
        set_u_ref(solver, zeros(1, N-1))
        
        status = solve(solver)
        # Status might be non-zero due to early termination, but should not crash
        @test status >= 0
    end
end 