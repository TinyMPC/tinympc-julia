# Interactive Cart-pole Animation (Julia)
# --------------------------------------
# Simulates the cart-pole using TinyMPC and creates a quick animation of the
# resulting motion.  The animation is saved as GIF next to this script.

include("../src/TinyMPC.jl")
using .TinyMPC
using LinearAlgebra
using Plots

# Use a non-interactive backend to work in CI/headless environments
ENV["GKSwstype"] = "100"

function simulate_cartpole(; steps=100)
    # Dynamics & MPC setup
    A = [1.0  0.01  0.0   0.0;
         0.0  1.0   0.039 0.0;
         0.0  0.0   1.002 0.01;
         0.0  0.0   0.458 1.002]
    B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
    Q = diagm([10.0, 1.0, 10.0, 1.0])
    R = diagm([1.0])
    ρ = 1.0; N = 10

    nx, nu = size(A,1), size(B,2)
    solver = TinyMPCSolver()
    setup!(solver, A, B, zeros(nx), Q, R, ρ, nx, nu, N, verbose=false)
    set_x_ref!(solver, zeros(nx, N))
    set_u_ref!(solver, zeros(nu, N-1))

    x = [0.3, 0.0, 0.0, 0.0]   # Initial state
    dt = 0.02                  # Time step for simple Euler integration

    xs = Vector{Vector{Float64}}(undef, steps)

    for k in 1:steps
        set_x0!(solver, x)
        solve!(solver)
        u = get_solution(solver).controls[1]
        # Simple discrete update (linear model)
        x = vec(A * x + B * u)
        xs[k] = copy(x)
    end
    return xs
end

function animate_cart(xs)
    # Extract states
    Nf = length(xs)
    cart_positions = [x[1] for x in xs]
    pole_angles     = [x[3] for x in xs]

    # Geometry
    cart_width  = 0.2
    pole_length = 1.0

    anim = @animate for k in 1:Nf
        cart_x = cart_positions[k]
        θ      = pole_angles[k]
        # Cart
        plot([cart_x - cart_width/2, cart_x + cart_width/2], [0, 0], lw=4, label="", xlims=(-1,1), ylims=(-0.1,1.2), aspect_ratio=:equal)
        # Pole (line from cart to tip)
        tip_x = cart_x + pole_length*sin(θ)
        tip_y = pole_length*cos(θ)
        plot!([cart_x, tip_x], [0, tip_y], lw=3, label="")
        scatter!([cart_x], [0], ms=8, label="")
        scatter!([tip_x], [tip_y], ms=6, label="")
        title!("Step $k")
    end
    gif(anim, joinpath(@__DIR__, "cartpole.gif"), fps=30)
    println("Animation saved to cartpole.gif")
end

if abspath(PROGRAM_FILE) == @__FILE__
    xs = simulate_cartpole(; steps=60)
    animate_cart(xs)
end 