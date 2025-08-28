# Cartpole Code Generation Example (Julia)
include("../src/TinyMPC.jl")
using .TinyMPC
using LinearAlgebra

# System matrices (matches Python/MATLAB examples)
A = [1.0  0.01  0.0   0.0;
     0.0  1.0   0.039 0.0;
     0.0  0.0   1.002 0.01;
     0.0  0.0   0.458 1.002]
B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
Q = diagm([10.0, 1.0, 10.0, 1.0])
R = diagm([1.0])
N = 20
rho = 1.0

# Input constraints (matching Python example)
u_min = fill(-0.5, 1, N-1)  # nu x (N-1)
u_max = fill(0.5, 1, N-1)   # nu x (N-1)

# Create solver and setup with bounds
solver = TinyMPCSolver()
status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)
@assert status == 0
set_bound_constraints(solver, fill(-Inf, 4, N), fill(Inf, 4, N), u_min, u_max)

# Set references (all zeros for code generation)
set_x_ref(solver, zeros(4, N))
set_u_ref(solver, zeros(1, N-1))

# Generate C++ code to ./out directory
out_dir = joinpath(@__DIR__, "out")
status = codegen(solver, out_dir, verbose=true)
@assert status == 0

println("Code generation completed successfully in: $out_dir") 