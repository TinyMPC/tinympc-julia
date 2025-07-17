using Test
using TinyMPC
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
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Compute sensitivity matrices using automatic differentiation
        sensitivities = compute_sensitivity_autograd(solver, A, B, Q, R, rho, verbose=false)
        
        # Check that we get the expected fields
        @test hasfield(typeof(sensitivities), :dK)
        @test hasfield(typeof(sensitivities), :dP)
        @test hasfield(typeof(sensitivities), :dC1)
        @test hasfield(typeof(sensitivities), :dC2)
        
        # Check dimensions
        @test size(sensitivities.dK) == (1, 4)  # nu x nx
        @test size(sensitivities.dP) == (4, 4)  # nx x nx
        @test size(sensitivities.dC1) == (1, 1)  # nu x nu
        @test size(sensitivities.dC2) == (4, 4)  # nx x nx
        
        # Check that matrices are finite
        @test all(isfinite.(sensitivities.dK))
        @test all(isfinite.(sensitivities.dP))
        @test all(isfinite.(sensitivities.dC1))
        @test all(isfinite.(sensitivities.dC2))
    end
    
    @testset "Set Sensitivity Matrices" begin
        solver = TinyMPCSolver()
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Create test sensitivity matrices
        dK = rand(1, 4)
        dP = rand(4, 4)
        dC1 = rand(1, 1)
        dC2 = rand(4, 4)
        
        # Set sensitivity matrices (should not error)
        @test_nowarn set_sensitivity_matrices!(solver, dK, dP, dC1, dC2, rho=rho, verbose=false)
    end
    
    @testset "Sensitivity Matrices for Adaptive Rho" begin
        solver = TinyMPCSolver()
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, 
                       adaptive_rho=true, verbose=false)
        @test status == 0
        
        # Compute sensitivities for adaptive rho
        sensitivities = compute_sensitivity_autograd(solver, A, B, Q, R, rho, verbose=false)
        
        # Set them in the solver
        set_sensitivity_matrices!(solver, sensitivities.dK, sensitivities.dP, 
                                 sensitivities.dC1, sensitivities.dC2, rho=rho, verbose=false)
        
        # Test that solver still works
        set_x0!(solver, [0.1, 0.0, 0.0, 0.0])
        set_x_ref!(solver, zeros(4, N))
        set_u_ref!(solver, zeros(1, N-1))
        
        status = solve!(solver)
        @test status == 0
    end
    
    @testset "Sensitivity Consistency" begin
        solver = TinyMPCSolver()
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Compute sensitivities twice with same parameters
        sens1 = compute_sensitivity_autograd(solver, A, B, Q, R, rho, verbose=false)
        sens2 = compute_sensitivity_autograd(solver, A, B, Q, R, rho, verbose=false)
        
        # Results should be identical (within numerical precision)
        @test sens1.dK ≈ sens2.dK
        @test sens1.dP ≈ sens2.dP
        @test sens1.dC1 ≈ sens2.dC1
        @test sens1.dC2 ≈ sens2.dC2
    end
    
    @testset "Different Rho Values" begin
        solver = TinyMPCSolver()
        status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
        @test status == 0
        
        # Compute sensitivities for different rho values
        sens_rho1 = compute_sensitivity_autograd(solver, A, B, Q, R, 0.5, verbose=false)
        sens_rho2 = compute_sensitivity_autograd(solver, A, B, Q, R, 2.0, verbose=false)
        
        # Results should be different for different rho values
        @test !(sens_rho1.dK ≈ sens_rho2.dK)
        @test !(sens_rho1.dP ≈ sens_rho2.dP)
        @test !(sens_rho1.dC1 ≈ sens_rho2.dC1)
        @test !(sens_rho1.dC2 ≈ sens_rho2.dC2)
    end
end 