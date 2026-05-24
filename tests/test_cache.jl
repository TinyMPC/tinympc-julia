using Test
isdefined(@__MODULE__, :TinyMPC) || include("../src/TinyMPC.jl")
using .TinyMPC
using LinearAlgebra

@testset "Cache Functionality" begin
    # Test system (cartpole)
    A = [1.0  0.01  0.0   0.0;
         0.0  1.0   0.039 0.0;
         0.0  0.0   1.002 0.01;
         0.0  0.0   0.458 1.002]
    B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
    Q = diagm([10.0, 1.0, 10.0, 1.0])
    R = diagm([1.0])
    N = 2  # Small horizon for fast testing
    rho = 1.0
    
    @testset "Set Cache Terms" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Create some test cache matrices
        Kinf = rand(1, 4)
        Pinf = rand(4, 4)
        Quu_inv = rand(1, 1)
        AmBKt = rand(4, 4)
        
        # Set cache terms (should not error)
        @test_nowarn set_cache_terms(solver, Kinf, Pinf, Quu_inv, AmBKt, verbose=false)
    end
    
    @testset "Set Computed Cache Terms" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0

        Kinf, Pinf, Quu_inv, AmBKt_raw = TinyMPC.solve_lqr(A, B, Q, R, rho)
        AmBKt = Matrix(AmBKt_raw)

        @test size(Kinf) == (1, 4)
        @test size(Pinf) == (4, 4)
        @test size(Quu_inv) == (1, 1)
        @test size(AmBKt) == (4, 4)
        @test all(isfinite.(Kinf))
        @test all(isfinite.(Pinf))
        @test all(isfinite.(Quu_inv))
        @test all(isfinite.(AmBKt))
        @test set_cache_terms(solver, Kinf, Pinf, Quu_inv, AmBKt, verbose=false) == 0
    end
end
