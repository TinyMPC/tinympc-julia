using Test
include("../src/TinyMPC.jl")
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
    
    @testset "Compute Cache Terms" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Compute cache terms
        cache = compute_cache_terms(solver, A, B, Q, R, rho=rho)
        
        # Check that cache contains expected fields
        @test hasfield(typeof(cache), :Kinf)
        @test hasfield(typeof(cache), :Pinf)
        @test hasfield(typeof(cache), :Quu_inv)
        @test hasfield(typeof(cache), :AmBKt)
        
        # Check dimensions
        @test size(cache.Kinf) == (1, 4)  # nu x nx
        @test size(cache.Pinf) == (4, 4)  # nx x nx
        @test size(cache.Quu_inv) == (1, 1)  # nu x nu
        @test size(cache.AmBKt) == (4, 4)  # nx x nx
        
        # Check that matrices are finite
        @test all(isfinite.(cache.Kinf))
        @test all(isfinite.(cache.Pinf))
        @test all(isfinite.(cache.Quu_inv))
        @test all(isfinite.(cache.AmBKt))
    end
    
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
    
    @testset "Cache Term Consistency" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Compute cache terms twice with same parameters
        cache1 = compute_cache_terms(solver, A, B, Q, R, rho=rho)
        cache2 = compute_cache_terms(solver, A, B, Q, R, rho=rho)
        
        # Results should be identical
        @test cache1.Kinf ≈ cache2.Kinf
        @test cache1.Pinf ≈ cache2.Pinf
        @test cache1.Quu_inv ≈ cache2.Quu_inv
        @test cache1.AmBKt ≈ cache2.AmBKt
    end
end 