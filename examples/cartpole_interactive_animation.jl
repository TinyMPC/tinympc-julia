# Cart-pole Swing-up Animation
# ------------------------------------

include("../src/TinyMPC.jl")
using .TinyMPC
using LinearAlgebra
using Plots
using Printf

# Non-interactive backend for headless environments
ENV["GKSwstype"] = "100"

function main()
    println("Creating cart-pole swing-up MPC animation...")
    
    # === SIMULATION SETUP ===
    max_steps = 500  # Maximum steps (20 seconds at 25 fps)
    
    # Enhanced system dynamics with very strong gravity - dramatic swing-up
    A = [1.0  0.01  0.0   0.0;
         0.0  1.0   0.065 0.0;    # Even more coupling from angle to cart velocity
         0.0  0.0   1.003 0.01;   # Increased angle integration 
         0.0  0.0   0.850 1.003]  # Very strong gravity effect (was 0.680)
    B = reshape([0.0; 0.045; 0.0; 0.120], 4, 1)  # Much increased control authority
    
    # MPC cost matrices for swing-up with stronger gravity
    Q = diagm([6.0, 1.5, 1.5, 4.0])  # Slightly lower penalties to allow faster motion
    R = diagm([0.6])  # Lower control penalty for more aggressive moves
    
    # Setup MPC solver for swing-up to inverted position
    solver = TinyMPCSolver()
    setup(solver, A, B, zeros(4), Q, R, 1.0, 4, 1, 20, verbose=false)
    
    # Reference: upright position (θ = 0 is upright, π is hanging down)
    x_ref = [0.0, 0.0, 0.0, 0.0]  # Cart at origin, pole upright
    set_x_ref(solver, repeat(x_ref, 1, 20))
    set_u_ref(solver, zeros(1, 19))
    
    # === RUN SIMULATION ===
    println("Finding optimal swing-up trajectory...")
    
    # Try multiple attempts to find the best swing-up trajectory
    best_states = nothing
    best_controls = nothing
    best_success_time = Inf
    
    for attempt in 1:5  # Try up to 5 attempts
        # Start with pole hanging straight down (θ = π) with slight variations
        variation = (attempt - 1) * 0.02  # Small variations between attempts
        x = [variation, 0.0, π + variation * 0.1, 0.0]  
        states = []
        controls = []
        
        # Track stability for early termination
        stable_count = 0
        required_stable_steps = 40  # Must be stable for 1.6 seconds
        
        println("  Attempt $(attempt)...")
        
        for k in 1:max_steps
        # Adaptive cost matrices - tuned for very strong gravity
        angle_error = abs(x[3])  # Distance from upright (0 rad)
        if angle_error < 0.5  # Close to upright
            # High penalty on angle deviation for stabilization
            Q_adaptive = diagm([15.0, 3.0, 300.0, 30.0])
        elseif angle_error < 1.0  # Intermediate region
            Q_adaptive = diagm([10.0, 2.0, 30.0, 15.0])
        else  # Far from upright (very aggressive swing-up phase)
            Q_adaptive = diagm([3.0, 0.8, 0.5, 0.8])  # Extremely low angle penalty for aggressive swing
        end
        
        # Update solver with adaptive costs every 10 steps for efficiency
        if k % 10 == 1
            setup(solver, A, B, zeros(4), Q_adaptive, R, 1.0, 4, 1, 20, verbose=false)
            set_x_ref(solver, repeat(x_ref, 1, 20))
            set_u_ref(solver, zeros(1, 19))
        end
        
        # MPC solve
        set_x0(solver, x)
        solve(solver, verbose=false)
        u = get_solution(solver).controls[1]
        
        # Store data and update state
        push!(states, copy(x))
        push!(controls, u)
        
        # Simple dynamics update
        x = vec(A * x + B * u)
        
        # Keep angle in [-π, π] range for proper visualization
        x[3] = mod(x[3] + π, 2π) - π
        
        # Check for stability (upright and not moving much)
        if abs(x[3]) < 0.1 && abs(x[4]) < 0.05 && abs(x[2]) < 0.05
            stable_count += 1
        else
            stable_count = 0  # Reset if not stable
        end
        
            # Early termination if stable for required duration
            if stable_count >= required_stable_steps
                success_time = k * 0.04
                println(@sprintf("    Success! Stabilized after %d steps (%.1f seconds)", k, success_time))
                
                # Check if this is the best attempt so far
                if success_time < best_success_time
                    best_success_time = success_time
                    best_states = copy(states)
                    best_controls = copy(controls)
                    println("    New best trajectory found!")
                end
                break
            end
        end
        
        # If this attempt succeeded, we have a good trajectory
        if length(states) > 0 && stable_count >= required_stable_steps
            break  # Stop trying more attempts
        end
    end
    
    # Use the best trajectory found
    if best_states === nothing
        error("Failed to find a successful swing-up trajectory after 5 attempts!")
    end
    
    states = best_states
    controls = best_controls
    final_steps = length(states)
    
    println(@sprintf("Using best trajectory: %.1f seconds", best_success_time))
    
    # === CREATE ANIMATION ===
    println("Creating beautiful swing-up animation...")
    create_animation(states, controls)
    println("Swing-up animation complete! Check cartpole.gif")
end

function create_animation(states, controls)
    """Create beautiful cart-pole swing-up animation with fancy visuals"""
    
    # Extract data
    positions = [s[1] for s in states]
    angles = [s[3] for s in states]
    N = length(states)
    
    # Animation parameters
    pole_length = 1.2  # Much longer pole for dramatic effect
    cart_width = 0.25
    cart_height = 0.12
    
    println(@sprintf("Creating %d frame animation (%.1f seconds)", N, N*0.04))
    
    anim = @animate for k in 1:N
        # Current state
        cart_x = positions[k]
        θ = angles[k]
        
        # Determine swing-up phase for visual effects
        phase = if abs(θ) > 2.0
            "SWING-UP"
        elseif abs(θ) > 0.5
            "APPROACHING"
        else
            "STABILIZING"
        end
        
        # === SETUP PLOT ===
        plot(size=(900, 700), dpi=120,
             xlims=(-6.0, 6.0), ylims=(-2.5, 2.5),
             aspect_ratio=:equal, background_color=:lightcyan,
             grid=false, showaxis=false)
        
        # === DRAW BEAUTIFUL GROUND ===
        plot!([-7, 7], [-0.12, -0.12], lw=12, color=:saddlebrown, label="")
        plot!([-7, 7], [-0.08, -0.08], lw=4, color=:forestgreen, label="")
        
        # === DRAW CART (realistic cart shape) ===
        # Cart body (rectangle)
        cart_corners_x = [cart_x - cart_width/2, cart_x + cart_width/2, 
                         cart_x + cart_width/2, cart_x - cart_width/2, cart_x - cart_width/2]
        cart_corners_y = [0, 0, cart_height, cart_height, 0]
        plot!(cart_corners_x, cart_corners_y, lw=3, fill=true, 
              color=:navy, fillcolor=:lightblue, label="")
        
        # Cart wheels
        wheel_radius = 0.04
        θ_circle = range(0, 2π, length=20)
        for wheel_offset in [-cart_width/3, cart_width/3]
            wheel_x = (cart_x + wheel_offset) .+ wheel_radius .* cos.(θ_circle)
            wheel_y = -0.02 .+ wheel_radius .* sin.(θ_circle)
            plot!(wheel_x, wheel_y, lw=2, fill=true, 
                  color=:black, fillcolor=:darkgray, label="")
        end
        
        # === DRAW POLE (enhanced longer pole) ===
        # Pole tip position (adjusted for cart height)
        tip_x = cart_x - pole_length * sin(θ)
        tip_y = cart_height/2 + pole_length * cos(θ)
        
        # Enhanced pole with gradient effect
        pole_color = if phase == "STABILIZING"
            :darkgreen
        elseif phase == "APPROACHING" 
            :darkorange
        else
            :darkred
        end
        
        # Thicker pole for better visibility
        plot!([cart_x, tip_x], [cart_height/2, tip_y], lw=8, color=pole_color, label="")
        
        # Pole tip mass (larger circle for longer pole)
        scatter!([tip_x], [tip_y], ms=15, color=:purple, 
                markerstroke=2, markerstrokecolor=:plum, label="")
        
        # Pivot point on cart
        scatter!([cart_x], [cart_height/2], ms=8, color=:orange, 
                markerstroke=1, markerstrokecolor=:darkorange, label="")
        
        # === TARGET STAR (beautiful golden star) ===
        target_x = cart_x - pole_length * sin(0)  # θ = 0 (upright)
        target_y = cart_height/2 + pole_length * cos(0)  # Adjusted for cart height
        scatter!([target_x], [target_y], ms=20, color=:gold, 
                markershape=:star, markerstroke=2, markerstrokecolor=:orange,
                alpha=0.9, label="")
        
        # === NO TRAJECTORY TRACE (clean view) ===
        
        # === BEAUTIFUL INFO DISPLAY ===
        title!("Cart-Pole Swing-Up Control with MPC", fontsize=18, fontweight=:bold)
        
        # Enhanced state display with colored backgrounds
        time_sec = k * 0.04
        annotate!((-3.5, 1.6), text(@sprintf("Time: %.1f s", time_sec), :left, 12, :black))
        annotate!((-3.5, 1.45), text(@sprintf("Angle: %.2f rad", θ), :left, 12, :black))
        annotate!((-3.5, 1.3), text(@sprintf("Position: %.2f m", cart_x), :left, 12, :black))
        
        # Phase indicator with color coding
        phase_color = if phase == "STABILIZING"
            :darkgreen
        elseif phase == "APPROACHING"
            :darkorange  
        else
            :darkred
        end
        annotate!((-3.5, 1.15), text(@sprintf("Phase: %s", phase), :left, 12, phase_color))
        
        # Progress indicator
        annotate!((3.5, 1.6), text(@sprintf("Frame: %d/%d", k, N), :right, 11, :gray))
        
        # Success celebration when stabilized
        if phase == "STABILIZING" && abs(θ) < 0.15
            annotate!((0, -1.6), text("✓ SUCCESSFULLY STABILIZED! ✓", :center, 16, :darkgreen))
            
            # Add some sparkle effects around the star
            sparkle_x = target_x .+ 0.1 .* [cos(k*0.5), cos(k*0.7+1), cos(k*0.3+2)]
            sparkle_y = target_y .+ 0.1 .* [sin(k*0.5), sin(k*0.7+1), sin(k*0.3+2)]
            scatter!(sparkle_x, sparkle_y, ms=6, color=:gold, alpha=0.7, label="")
        end
    end
    
    # Save with dynamic duration
    duration = N * 0.04
    println(@sprintf("Saving swing-up animation (%.1f seconds)...", duration))
    gif(anim, joinpath(@__DIR__, "cartpole.gif"), fps=25)
    println(@sprintf("Animation saved to cartpole.gif (%d frames, %.1f seconds)", N, duration))
end

# Run the animation
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end