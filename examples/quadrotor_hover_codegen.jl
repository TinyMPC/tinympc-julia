#!/usr/bin/env julia
# Quadrotor hover with sensitivity-enabled code generation
# Matches Python/MATLAB quadrotor examples by computing ∂{K,P,C1,C2}/∂ρ
# then generating C++ code with these sensitivity matrices.

using ForwardDiff
using LinearAlgebra
using Random

Random.seed!(1)

# Load TinyMPC
include("../src/TinyMPC.jl")
using .TinyMPC

# ------------------------- Problem definition --------------------------------
# We keep parameters local to avoid constant re-definitions when this file is
# `include`-d multiple times during the test suite.
ρ = 5.0               # ADMM penalty parameter
N  = 20               # Horizon length

# Toggle switch for adaptive rho
ENABLE_ADAPTIVE_RHO = false # Set to false to disable adaptive rho

# State-space matrices
A = [
    1.0 0.0 0.0 0.0 0.024525 0.0 0.05 0.0 0.0 0.0 0.0002044 0.0;
    0.0 1.0 0.0 -0.024525 0.0 0.0 0.0 0.05 0.0 -0.0002044 0.0 0.0;
    0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.05 0.0 0.0 0.0;
    0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.025 0.0 0.0;
    0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.025 0.0;
    0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.025;
    0.0 0.0 0.0 0.0 0.981 0.0 1.0 0.0 0.0 0.0 0.0122625 0.0;
    0.0 0.0 0.0 -0.981 0.0 0.0 0.0 1.0 0.0 -0.0122625 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
]

B = [
    -0.0007069  0.0007773  0.0007091 -0.0007795;
     0.0007034  0.0007747 -0.0007042 -0.0007739;
     0.0052554  0.0052554  0.0052554  0.0052554;
    -0.1720966 -0.1895213  0.1722891  0.1893288;
    -0.1729419  0.190174   0.1734809 -0.1907131;
     0.0123423 -0.0045148 -0.0174024  0.0095748;
    -0.056552   0.0621869  0.0567283 -0.0623632;
     0.0562756  0.0619735 -0.0563386 -0.0619105;
     0.2102143  0.2102143  0.2102143  0.2102143;
   -13.7677303 -15.1617018 13.7831318 15.1463003;
   -13.8353509 15.2139209 13.8784751 -15.2570451;
     0.9873856 -0.361182  -1.392188   0.7659845
]

Q = diagm([100.0, 100.0, 100.0, 4.0, 4.0, 400.0,
                 4.0, 4.0, 4.0, 2.0408163, 2.0408163, 4.0])
R = diagm([4.0, 4.0, 4.0, 4.0])

nx = size(A, 1)
nu = size(B, 2)
f  = zeros(nx)          # No affine term

# ------------------------- Main workflow --------------------------------------
function main()
    # 1) Create and set-up solver
    solver = TinyMPCSolver()
    @assert setup(solver, A, B, f, Q, R, ρ, nx, nu, N, 
                 verbose=true, adaptive_rho=ENABLE_ADAPTIVE_RHO) == 0

    # 2) Hover reference (all zeros)
    set_x_ref(solver, zeros(nx, N))
    set_u_ref(solver, zeros(nu, N-1))

    # Output directory for code generation
    out_dir = joinpath(@__DIR__, "out")
    mkpath(out_dir)

    if ENABLE_ADAPTIVE_RHO
        println("Enabled adaptive rho - generating code with sensitivity matrices...")
        
        # 3) Compute cache terms and sensitivities using built-in function
        @info "Computing cache terms and sensitivities (finite-difference)…"
        dK, dP, dC1, dC2 = compute_sensitivity_autograd(solver)

        # 4) Code generation with sensitivity matrices
        # The sensitivity matrices are passed directly to codegen_with_sensitivity
        codegen_with_sensitivity(solver, out_dir, dK, dP, dC1, dC2; verbose=true)
    else
        println("Running without adaptive rho - generating code without sensitivity matrices...")
        
        # 4) Regular code generation without sensitivity
        codegen(solver, out_dir, verbose=true)
    end
    
    println("Code generation completed successfully in: $(out_dir)")
end

# Execute when run as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 