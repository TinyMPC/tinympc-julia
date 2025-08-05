# Cartpole MPC Example (Julia)
include("../src/TinyMPC.jl")
using .TinyMPC
using LinearAlgebra

function main()
    # Cartpole system matrices (matches Python/MATLAB examples)
    A = [1.0  0.01  0.0   0.0;
         0.0  1.0   0.039 0.0;
         0.0  0.0   1.002 0.01;
         0.0  0.0   0.458 1.002]
    B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
    Q = diagm([10.0, 1.0, 10.0, 1.0])
    R = diagm([1.0])
    N = 20
    rho = 1.0

    # Create solver and setup
    solver = TinyMPCSolver()
    status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, 
                   max_iter=10, verbose=false)
    @assert status == 0 "Solver setup failed"

    # Initial state and references
    x0 = [0.5, 0.0, 0.0, 0.0]
    set_x0(solver, x0)
    set_x_ref(solver, zeros(4, N))
    set_u_ref(solver, zeros(1, N-1))

    # Simulation loop (matching Python example duration)
    Nsim = 200
    xs = Matrix{Float64}(undef, 4, Nsim)
    us = Vector{Float64}(undef, Nsim)

    for k in 1:Nsim
        # Solve MPC problem
        status = solve(solver)
        @assert status == 0 "Solve failed at step $k"
        
        # Get solution and extract first control
        sol = get_solution(solver)
        u = sol.controls[1]
        
        # Simulate forward (same as Python: x = A@x + B@u)
        x0 = vec(A * x0 + B * u)
        set_x0(solver, x0)
        
        # Store results
        xs[:, k] = x0
        us[k] = u
    end

    println("MPC simulation completed successfully")
    println("Final state: ", xs[:, end])
    println("Average control effort: ", sum(abs.(us))/length(us))
    
    # Note: Plotting requires Plots.jl - uncomment if available
    # using Plots
    # p1 = plot(xs', label=["x (meters)" "θ (radians)" "ẋ (m/s)" "θ̇ (rad/s)"], 
    #           title="Cartpole trajectory over time", xlabel="time steps")
    # p2 = plot(us, label="control (Newtons)", title="Control Input", xlabel="time steps")
    # plot(p1, p2, layout=(2,1))
end

main() 