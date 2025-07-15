# Cartpole MPC Example (Julia)
using TinyMPC, LinearAlgebra, Plots

function main()

    # Cartpole system matrices
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
    f = zeros(4)
    status = setup!(solver, A, B, f, Q, R, rho, 4, 1, N, verbose=false)
    @assert status == 0

    # Initial state and references
    x = [0.5, 0.0, 0.0, 0.0]
    set_x0!(solver, x)
    set_x_ref!(solver, zeros(4, N))
    set_u_ref!(solver, zeros(1, N-1))

    # Simulation loop
    Nsim = 200
    xs = Matrix{Float64}(undef, 4, Nsim)
    us = Vector{Float64}(undef, Nsim)

    for k in 1:Nsim
        solve!(solver)
        sol = get_solution(solver)
        u = sol.controls[1]
        x = vec(A * x + B * u)
        set_x0!(solver, x)
        xs[:, k] = x
        us[k] = u
    end

    plot(xs', label=["x" "θ" "ẋ" "θ̇"], layout=(2,1), title="Cartpole States")
    plot!(us, subplot=2, label="u", title="Control Input")
end

main() 