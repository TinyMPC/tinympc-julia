# Cart-pole One-Solve Demo (Julia)
# --------------------------------
# Runs TinyMPC once for a classic 4-state cart-pole system and prints the
# optimal first control input plus resulting predicted trajectory.

using TinyMPC
using LinearAlgebra

# System definition (matches other cart-pole examples)
A = [1.0  0.01  0.0   0.0;
     0.0  1.0   0.039 0.0;
     0.0  0.0   1.002 0.01;
     0.0  0.0   0.458 1.002]
B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
Q = diagm([10.0, 1.0, 10.0, 1.0])
R = diagm([1.0])
N = 20
ρ = 1.0

nx, nu = size(A, 1), size(B, 2)
solver = TinyMPCSolver()
setup!(solver, A, B, zeros(nx), Q, R, ρ, nx, nu, N, verbose=false)

# Initial disturbance
x0 = [0.3, 0.0, 0.0, 0.0]
set_x0!(solver, x0)
set_x_ref!(solver, zeros(nx, N))
set_u_ref!(solver, zeros(nu, N-1))

solve!(solver)
sol = get_solution(solver)

println("First optimal control u₀ = ", sol.controls[1])
println("Predicted next state x₁ = ", sol.states[:, 2]) 