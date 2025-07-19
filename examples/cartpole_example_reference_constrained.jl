# Cartpole MPC with state/input box constraints (Julia)
using TinyMPC, LinearAlgebra, Plots

function main()
    # Dynamics
    A = [1.0 0.01 0 0;
         0 1.0 0.039 0;
         0 0 1.002 0.01;
         0 0 0.458 1.002]
    B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
    Q = diagm([10.0, 1.0, 10.0, 1.0])
    R = diagm([1.0])
    N = 20; rho = 1.0

    # Constraints (cart position ±2 m, force ±5 N)
    x_min = fill(-Inf, 4, N); x_max = fill(Inf, 4, N);
    x_min[1, :] .= -2; x_max[1, :] .= 2  # cart position limit
    u_min = fill(-5.0, 1, N-1); u_max = fill(5.0, 1, N-1)

    solver = TinyMPCSolver()
    status = setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N)
    @assert status == 0
    set_bound_constraints(solver, x_min, x_max, u_min, u_max)

    x = [0.0, 0, 0.1, 0]  # small angle perturbation
    set_x0(solver, x)
    set_x_ref(solver, zeros(4, N))
    set_u_ref(solver, zeros(1, N-1))

    Nsim = 150; xs = zeros(4, Nsim); us = zeros(Nsim)
    for k in 1:Nsim
        solve(solver)
        sol = get_solution(solver)
        u = sol.controls[1]
        x = vec(A * x + B * u)
        set_x0(solver, x)
        xs[:, k] = x; us[k] = u
    end

    plot(xs', layout=(2,1), label=["x" "θ" "ẋ" "θ̇"], title="States with constraints")
    plot!(us, subplot=2, label="u", title="Control (bounded ±5 N)")
end

main() 