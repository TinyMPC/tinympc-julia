using Test
isdefined(@__MODULE__, :TinyMPC) || include("../src/TinyMPC.jl")
using .TinyMPC
using LinearAlgebra

@testset "Sensitivity Functionality" begin
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
    
    @testset "Compute Sensitivity AutoGrad" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Compute sensitivity matrices using finite differences.
        dK, dP, dC1, dC2 = compute_sensitivity_autograd(solver)
        
        # Check dimensions
        @test size(dK) == (1, 4)  # nu x nx
        @test size(dP) == (4, 4)  # nx x nx
        @test size(dC1) == (1, 1)  # nu x nu
        @test size(dC2) == (4, 4)  # nx x nx
        
        # Check that matrices are finite
        @test all(isfinite.(dK))
        @test all(isfinite.(dP))
        @test all(isfinite.(dC1))
        @test all(isfinite.(dC2))
    end
    
    @testset "Sensitivity Code Generation" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, 
                       adaptive_rho=true, verbose=false)
        @test status == 0
        
        dK, dP, dC1, dC2 = compute_sensitivity_autograd(solver)
        set_x_ref(solver, zeros(4, N))
        set_u_ref(solver, zeros(1, N-1))

        out_dir = mktempdir()
        @test codegen_with_sensitivity(solver, out_dir, dK, dP, dC1, dC2, verbose=false) == 0
        @test isfile(joinpath(out_dir, "src", "tiny_data.cpp"))
        rm(out_dir, recursive=true)
    end
    
    @testset "Sensitivity Consistency" begin
        solver = TinyMPCSolver()
        status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Compute sensitivities twice with same parameters
        sens1 = compute_sensitivity_autograd(solver)
        sens2 = compute_sensitivity_autograd(solver)
        
        # Results should be identical (within numerical precision)
        @test sens1[1] ≈ sens2[1]
        @test sens1[2] ≈ sens2[2]
        @test sens1[3] ≈ sens2[3]
        @test sens1[4] ≈ sens2[4]
    end
    
    @testset "Different Rho Values" begin
        solver_rho1 = TinyMPCSolver()
        solver_rho2 = TinyMPCSolver()
        @test setup(solver_rho1, A, B, zeros(4), Q, R, 0.5, 4, 1, N, verbose=false) == 0
        @test setup(solver_rho2, A, B, zeros(4), Q, R, 2.0, 4, 1, N, verbose=false) == 0

        sens_rho1 = compute_sensitivity_autograd(solver_rho1)
        sens_rho2 = compute_sensitivity_autograd(solver_rho2)
        
        # Results should be different for different rho values
        @test !(sens_rho1[1] ≈ sens_rho2[1])
        @test !(sens_rho1[2] ≈ sens_rho2[2])
        @test !(sens_rho1[3] ≈ sens_rho2[3])
        @test !(sens_rho1[4] ≈ sens_rho2[4])
    end
end
