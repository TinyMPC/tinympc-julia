"""
Rocket Landing with Constraints
Based on: https://github.com/TinyMPC/TinyMPC/blob/main/examples/rocket_landing_mpc.cpp
"""

using TinyMPC
using LinearAlgebra
using Plots
using Printf

# Problem dimensions
const NSTATES = 6  # [x, y, z, vx, vy, vz] 
const NINPUTS = 3  # [thrust_x, thrust_y, thrust_z]
const NHORIZON = 10

# System dynamics (from rocket_landing_params_20hz.hpp)
A = [1.0  0.0  0.0  0.05  0.0   0.0;
     0.0  1.0  0.0  0.0   0.05  0.0;
     0.0  0.0  1.0  0.0   0.0   0.05;
     0.0  0.0  0.0  1.0   0.0   0.0;
     0.0  0.0  0.0  0.0   1.0   0.0;
     0.0  0.0  0.0  0.0   0.0   1.0]

B = [0.000125  0.0       0.0;
     0.0       0.000125  0.0;
     0.0       0.0       0.000125;
     0.005     0.0       0.0;
     0.0       0.005     0.0;
     0.0       0.0       0.005]

fdyn = [0.0, 0.0, -0.0122625, 0.0, 0.0, -0.4905]
Q = diagm([101.0, 101.0, 101.0, 101.0, 101.0, 101.0])  # From parameter file
R = diagm([2.0, 2.0, 2.0])  # From parameter file

# Box constraints
x_min = fill(-1e17, NSTATES, NHORIZON)
x_max = fill(1e17, NSTATES, NHORIZON)
u_min = fill(-1e17, NINPUTS, NHORIZON-1)
u_max = fill(1e17, NINPUTS, NHORIZON-1)

# Apply specific bounds
x_min[1, :] .= -5.0; x_max[1, :] .= 5.0      # x position
x_min[2, :] .= -5.0; x_max[2, :] .= 5.0      # y position  
x_min[3, :] .= -0.5; x_max[3, :] .= 100.0    # z position
x_min[4, :] .= -10.0; x_max[4, :] .= 10.0    # x velocity
x_min[5, :] .= -10.0; x_max[5, :] .= 10.0    # y velocity
x_min[6, :] .= -20.0; x_max[6, :] .= 20.0    # z velocity

u_min .= -10.0; u_max .= 105.0               # thrust limits

# SOC constraints 
cx = [0.5]     # coefficients for state cones (mu)
cu = [0.25]    # coefficients for input cones (mu)
Acx = [0]      # start indices for state cones (0-indexed for C++)
Acu = [0]      # start indices for input cones (0-indexed for C++)
qcx = [3]      # dimensions for state cones
qcu = [3]      # dimensions for input cones

# Setup solver
solver = TinyMPCSolver()
setup(solver, A, B, fdyn, Q, R, 1.0, NSTATES, NINPUTS, NHORIZON,
      verbose=true, max_iter=100, abs_pri_tol=2e-3, abs_dua_tol=1e-3)

# Set bound constraints explicitly, auto-enables flags
set_bound_constraints(solver, x_min, x_max, u_min, u_max)

# Set cone constraints (inputs first), auto-enables flags
set_cone_constraints(solver, Int32.(Acu), Int32.(qcu), cu, Int32.(Acx), Int32.(qcx), cx, verbose=true)

# Initial and goal states
xinit = [4.0, 2.0, 20.0, -3.0, 2.0, -4.5]
xgoal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Initial reference trajectory (will be updated each step like C++)
x_ref = zeros(NSTATES, NHORIZON)
u_ref = zeros(NINPUTS, NHORIZON-1)

# Animation setup - Extended for longer simulation
const NTOTAL = 100  # Match C++
x_current = xinit * 1.1  # Match C++ (x0 = xinit * 1.1)

# Set initial reference
for i in 1:NHORIZON
    x_ref[:, i] = xinit + (xgoal - xinit) * (i-1) / (NTOTAL - 1)  # Use NTOTAL-1 like C++
end
u_ref[3, :] .= 10.0  # Hover thrust

set_x_ref(solver, x_ref)
set_u_ref(solver, u_ref)

# Store trajectory for plotting
trajectory = []
controls = []
constraint_violations = []

println("Starting rocket landing simulation...")
for k in 1:(NTOTAL - NHORIZON)
    global x_current  # Declare as global to modify in loop
    
    # Calculate tracking error (match C++ exactly: (x0 - Xref.col(1)).norm())
    tracking_error = norm(x_current - x_ref[:, 2])
    @printf("tracking error: %.5f\n", tracking_error)
    
    # 1. Update measurement (set current state)
    set_x0(solver, x_current)
    
    # 2. Update reference trajectory (match C++ exactly)
    for i in 1:NHORIZON
        x_ref[:, i] = xinit + (xgoal - xinit) * (i + k - 2) / (NTOTAL - 1)
        if i <= NHORIZON - 1
            u_ref[3, i] = 10.0  # uref stays constant
        end
    end
    
    set_x_ref(solver, x_ref)
    set_u_ref(solver, u_ref)
    
    # 3. Solve MPC problem
    status = solve(solver)
    solution = get_solution(solver)
    
    # 4. Simulate forward (apply first control)
    u_opt = solution.controls[:, 1]
    x_current = A * x_current + B * u_opt + fdyn
    
    # Store data for plotting
    push!(trajectory, copy(x_current))
    push!(controls, copy(u_opt))
    
    # Check constraint violations
    altitude_violation = x_current[3] < 0  # Ground constraint
    thrust_violation = norm(u_opt[1:2]) > 0.25 * abs(u_opt[3])  # Cone constraint
    push!(constraint_violations, altitude_violation || thrust_violation)
end

# Convert to matrices for plotting
traj_matrix = hcat(trajectory...)
ctrl_matrix = hcat(controls...)

println("\nSimulation completed!")
@printf("Initial state was: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", (xinit * 1.1)...)
@printf("Final position: [%.2f, %.2f, %.2f]\n", x_current[1:3]...)
@printf("Final velocity: [%.2f, %.2f, %.2f]\n", x_current[4:6]...)
@printf("Distance to goal: %.3f m\n", norm(x_current[1:3]))
@printf("Constraint violations: %d/%d\n", sum(constraint_violations), length(constraint_violations))

# Plotting
time_steps = 1:size(traj_matrix, 2)

p1 = plot(traj_matrix[1, :], traj_matrix[2, :], lw=2, label="Trajectory", 
          title="2D Trajectory (X-Y)", xlabel="X (m)", ylabel="Y (m)")
scatter!(p1, [(xinit * 1.1)[1]], [(xinit * 1.1)[2]], c=:red, ms=8, label="Start")
scatter!(p1, [xgoal[1]], [xgoal[2]], c=:green, ms=8, label="Goal")

p2 = plot(time_steps, traj_matrix[1, :], label="X", lw=1.5, c=:red,
          title="Position vs Time", xlabel="Time Step", ylabel="Position (m)")
plot!(p2, time_steps, traj_matrix[2, :], label="Y", lw=1.5, c=:green)
plot!(p2, time_steps, traj_matrix[3, :], label="Z", lw=1.5, c=:blue)
hline!(p2, [0], ls=:dash, c=:black, alpha=0.5, label="Ground")

p3 = plot(time_steps, traj_matrix[4, :], label="Vx", lw=1.5, c=:red,
          title="Velocity vs Time", xlabel="Time Step", ylabel="Velocity (m/s)")
plot!(p3, time_steps, traj_matrix[5, :], label="Vy", lw=1.5, c=:green)
plot!(p3, time_steps, traj_matrix[6, :], label="Vz", lw=1.5, c=:blue)

p4 = plot(time_steps, ctrl_matrix[1, :], label="Thrust X", lw=1.5, c=:red,
          title="Thrust vs Time", xlabel="Time Step", ylabel="Thrust (N)")
plot!(p4, time_steps, ctrl_matrix[2, :], label="Thrust Y", lw=1.5, c=:green)
plot!(p4, time_steps, ctrl_matrix[3, :], label="Thrust Z", lw=1.5, c=:blue)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900))
