# Cart-pole One-Solve Example (Julia)
# ----------------------------------------
# Simple example demonstrating a single MPC solve for a cart-pole system
# Matches functionality of cartpole_example_one_solve.py

include("../src/TinyMPC.jl")
using .TinyMPC
using LinearAlgebra

# System matrices (matches Python example exactly)
A = [1.0  0.01  0.0   0.0;
     0.0  1.0   0.039 0.0;
     0.0  0.0   1.002 0.01;
     0.0  0.0   0.458 1.002]
B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
Q = diagm([10.0, 1.0, 10.0, 1.0])
R = diagm([1.0])

N = 20

# Create solver and setup
prob = TinyMPCSolver()
setup(prob, A, B, zeros(4), Q, R, 1.0, 4, 1, N, max_iter=10)

# Set initial state
x0 = [0.5, 0.0, 0.0, 0.0]
set_x0(prob, x0)

# Solve and print results
solve(prob)
solution = get_solution(prob)
println(solution.controls)