# Cartpole Code Generation Example (Julia)
using TinyMPC, LinearAlgebra

# System matrices
A = [1.0  0.01  0.0   0.0;
     0.0  1.0   0.039 0.0;
     0.0  0.0   1.002 0.01;
     0.0  0.0   0.458 1.002]
B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
Q = diagm([10.0, 1.0, 10.0, 1.0])
R = diagm([1.0])
N = 20
rho = 1.0

solver = TinyMPCSolver()
status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N)
@assert status == 0

set_x_ref!(solver, zeros(4, N))
set_u_ref!(solver, zeros(1, N-1))

# Generate C++ code to ./out directory
codegen(solver, joinpath(@__DIR__, "out"), verbose=true)
println("✅ Code generation complete") 