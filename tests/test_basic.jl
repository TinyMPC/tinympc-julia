include("../src/TinyMPC.jl")
using .TinyMPC
using Test
using LinearAlgebra

@testset "Basic TinyMPC Functionality" begin
    # Test system (cartpole)
    A = [1.0  0.01  0.0   0.0;
         0.0  1.0   0.039 0.0;
         0.0  0.0   1.002 0.01;
         0.0  0.0   0.458 1.002]
    B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
    Q = diagm([10.0, 1.0, 10.0, 1.0])
    R = diagm([1.0])
    N = 10
    rho = 1.0
    
    @testset "Solver Setup" begin
        solver = TinyMPCSolver()
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
    end
    
    @testset "Reference Setting and Solving" begin
        solver = TinyMPCSolver()
        setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        
        # Set initial state
        x0 = [0.5, 0.0, 0.0, 0.0]
        set_x0!(solver, x0)
        
        # Set references
        set_x_ref!(solver, zeros(4, N))
        set_u_ref!(solver, zeros(1, N-1))
        
        # Solve
        status = solve!(solver)
        @test status == 0
        
        # Get solution
        sol = get_solution(solver)
        @test size(sol.states) == (4, N)
        @test size(sol.controls) == (1, N-1)
        @test length(sol.controls) == N-1
    end
    
    @testset "Bounds Constraints" begin
        solver = TinyMPCSolver()
        u_min = fill(-1.0, 1, N-1)
        u_max = fill(1.0, 1, N-1)
        
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, 
                       u_min=u_min, u_max=u_max, verbose=false)
        @test status == 0
        
        # Verify constraints are respected
        set_x0!(solver, [1.0, 0.0, 0.0, 0.0])  # Large initial disturbance
        set_x_ref!(solver, zeros(4, N))
        set_u_ref!(solver, zeros(1, N-1))
        
        status = solve!(solver)
        @test status == 0
        
        sol = get_solution(solver)
        @test all(sol.controls .>= -1.0)
        @test all(sol.controls .<= 1.0)
    end
end 