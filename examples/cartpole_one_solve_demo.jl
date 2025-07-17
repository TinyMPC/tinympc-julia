# Cart-pole One-Solve Demo (Julia)
# --------------------------------
# Runs TinyMPC once for a classic 4-state cart-pole system and prints the
# optimal first control input plus resulting predicted trajectory.
# Matches Python cartpole_example_one_solve.py functionality.

include("../src/TinyMPC.jl")
using .TinyMPC
using LinearAlgebra

# System definition (matches Python/MATLAB examples exactly)
A = [1.0  0.01  0.0   0.0;
     0.0  1.0   0.039 0.0;
     0.0  0.0   1.002 0.01;
     0.0  0.0   0.458 1.002]
B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
Q = diagm([10.0, 1.0, 10.0, 1.0])
R = diagm([1.0])
N = 20
rho = 1.0

# Problem dimensions
nx, nu = size(A, 1), size(B, 2)

# Create and setup solver
solver = TinyMPCSolver()
status = setup!(solver, A, B, zeros(nx), Q, R, rho, nx, nu, N, 
               max_iter=10, verbose=false)
@assert status == 0 "Solver setup failed"

# Initial state disturbance (matching Python example)
x0 = [0.5, 0.0, 0.0, 0.0]  # Changed to match Python value
set_x0!(solver, x0)
set_x_ref!(solver, zeros(nx, N))
set_u_ref!(solver, zeros(nu, N-1))

# Solve once
status = solve!(solver)
@assert status == 0 "Solve failed"

# Get and display solution
sol = get_solution(solver)
println("First optimal control u₀ = ", sol.controls[1])
println("Full control sequence: ", sol.controls)

# Optional: Show predicted trajectory
println("Initial state: ", x0)
println("Predicted next state x₁ = ", sol.states[:, 2]) 