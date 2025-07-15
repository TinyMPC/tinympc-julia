# Simple TinyMPC demonstration
# This script shows basic usage of the TinyMPC Julia wrapper

using TinyMPC
using LinearAlgebra

println("TinyMPC Julia Wrapper Demo")
println("==========================")

# Define a simple cartpole system
println("\n1. Setting up cartpole system...")

# System matrices
A = [1.0  0.01  0.0   0.0;
     0.0  1.0   0.039 0.0;
     0.0  0.0   1.002 0.01;
     0.0  0.0   0.458 1.002]

B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
f = zeros(4)  # No affine dynamics

# Cost matrices
Q = diagm([10.0, 1.0, 10.0, 1.0])  # State cost
R = reshape([1.0], 1, 1)           # Input cost

# Problem dimensions
nx = 4   # States: [position, velocity, angle, angular_velocity]
nu = 1   # Input: force
N = 10   # Prediction horizon
rho = 1.0  # ADMM penalty parameter

println("   States: $nx, Inputs: $nu, Horizon: $N")

# Create and setup solver
println("\n2. Creating TinyMPC solver...")
solver = TinyMPCSolver()

println("   Calling setup...")
status = setup!(solver, A, B, f, Q, R, rho, nx, nu, N, verbose=false)
if status == 0
    println("   ✓ Setup successful")
else
    println("   ✗ Setup failed with status: $status")
    exit(1)
end

# Set initial state (perturbed from equilibrium)
println("\n3. Setting initial conditions...")
x0 = [0.1, 0.0, 0.1, 0.0]  # Small position and angle disturbance
status = set_x0!(solver, x0)
println("   Initial state: $x0")

# Set reference trajectory (stabilize at origin)
x_ref = zeros(nx, N)
u_ref = zeros(nu, N-1)
set_x_ref!(solver, x_ref)
set_u_ref!(solver, u_ref)
println("   Reference: stabilize at origin")

# Set loose box constraints
x_min = fill(-10.0, nx, N)
x_max = fill(10.0, nx, N)
u_min = fill(-5.0, nu, N-1)
u_max = fill(5.0, nu, N-1)
set_bound_constraints!(solver, x_min, x_max, u_min, u_max)

# Configure solver settings
update_settings!(solver, max_iter=100, abs_pri_tol=1e-3, abs_dua_tol=1e-3)

# Solve the MPC problem
println("\n4. Solving MPC problem...")
status = solve!(solver)

if status == 0
    println("   ✓ Solve completed successfully")
else
    println("   ⚠ Solve completed with status: $status")
end

# Get and display results
println("\n5. Results:")
solution = get_solution(solver)
states = solution.states
controls = solution.controls

   println("   Iterations: $(get_iterations(solver))")
   println("   Solved: $(is_solved(solver))")
println("   Final state: $(states[:, end])")
println("   First control input: $(controls[:, 1])")

# Show trajectory evolution
println("\n6. State trajectory:")
println("   Time step | Position | Velocity |  Angle   | Ang.Vel  | Control")
println("   --------- | -------- | -------- | -------- | -------- | -------")
for k in 1:min(5, N)
    pos, vel, ang, angvel = states[:, k]
    control = k <= N-1 ? controls[1, k] : 0.0
    println("      $k      | $(round(pos, digits=3)) | $(round(vel, digits=3)) | $(round(ang, digits=3)) | $(round(angvel, digits=3)) | $(round(control, digits=3))")
end

# Basic performance check
initial_energy = 0.5 * (x0[1]^2 + x0[3]^2)  # Simple energy measure
final_energy = 0.5 * (states[1, end]^2 + states[3, end]^2)

println("\n7. Performance:")
println("   Initial energy: $(round(initial_energy, digits=4))")
println("   Final energy: $(round(final_energy, digits=4))")

if final_energy < initial_energy
    println("   ✓ System is stabilizing")
else
    println("   ⚠ System may not be fully stabilized (might need more iterations)")
end

println("\n✓ Demo completed successfully!")
println("You can now use TinyMPC in your Julia projects.") 