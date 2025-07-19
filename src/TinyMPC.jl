module TinyMPC

export TinyMPCSolver, setup, solve, get_solution, set_x0, set_x_ref, set_u_ref, 
       set_bound_constraints, update_settings, compute_cache_terms,
       set_sensitivity_matrices, set_cache_terms, print_problem_data,
       compute_sensitivity_autograd, codegen, codegen_with_sensitivity

using LinearAlgebra, Libdl

# Runtime state management (simplified)
const _state = Ref{Dict{Symbol, Any}}()
_init_state() = isassigned(_state) ? _state[] : (_state[] = Dict(:lib_loaded => false, :lib_path => joinpath(dirname(@__DIR__), "lib", "libtinympc_jl")))
_lib_path() = _init_state()[:lib_path] * "." * Libdl.dlext
_ensure_loaded() = _init_state()[:lib_loaded] || (isfile(_lib_path()) ? (_init_state()[:lib_loaded] = true) : error("TinyMPC library not found"))

# Empty init to avoid precompilation issues
__init__() = nothing

"""
    TinyMPCSolver

A lightweight MPC solver instance. Use `setup` to configure the problem
and `solve` to compute optimal trajectories.

# Example
```julia
solver = TinyMPCSolver()
setup(solver, A, B, f, Q, R, rho, nx, nu, N)
set_x0(solver, x0)
solve(solver)
solution = get_solution(solver)
```
"""
struct TinyMPCSolver
    nx::Ref{Int}
    nu::Ref{Int}
    N::Ref{Int}
    rho::Ref{Float64}
    is_setup::Ref{Bool}
    A::Ref{Matrix{Float64}}
    B::Ref{Matrix{Float64}}
    Q::Ref{Matrix{Float64}}
    R::Ref{Matrix{Float64}}
    
    TinyMPCSolver() = new(Ref(0), Ref(0), Ref(0), Ref(0.0), Ref(false), 
                         Ref(Matrix{Float64}(undef, 0, 0)), Ref(Matrix{Float64}(undef, 0, 0)),
                         Ref(Matrix{Float64}(undef, 0, 0)), Ref(Matrix{Float64}(undef, 0, 0)))
end

"""
    setup(solver, A, B, f, Q, R, rho, nx, nu, N; kwargs...)

Setup the MPC problem with system matrices and parameters.
"""
function setup(solver::TinyMPCSolver, A::Matrix{Float64}, B::Matrix{Float64}, f::Vector{Float64},
               Q::Matrix{Float64}, R::Matrix{Float64}, rho::Float64, nx::Int, nu::Int, N::Int;
               verbose::Bool=false, abs_pri_tol::Float64=1e-3, abs_dua_tol::Float64=1e-3,
               max_iter::Int=100, check_termination::Bool=true, enable_state_bound::Bool=false,
               enable_input_bound::Bool=false, adaptive_rho::Bool=false,
               adaptive_rho_min::Float64=0.1, adaptive_rho_max::Float64=10.0,
               adaptive_rho_clipping::Bool=true,
               x_min::Union{Matrix{Float64}, Nothing}=nothing, x_max::Union{Matrix{Float64}, Nothing}=nothing,
               u_min::Union{Matrix{Float64}, Nothing}=nothing, u_max::Union{Matrix{Float64}, Nothing}=nothing)
    
    _ensure_loaded()
    
    # Set default bounds if not provided
    x_min = x_min === nothing ? fill(-1e17, nx, N) : x_min
    x_max = x_max === nothing ? fill(1e17, nx, N) : x_max
    u_min = u_min === nothing ? fill(-1e17, nu, N-1) : u_min
    u_max = u_max === nothing ? fill(1e17, nu, N-1) : u_max
    
    f_matrix = reshape(f, nx, 1)
    
    # Store problem data in solver
    solver.nx[] = nx
    solver.nu[] = nu
    solver.N[] = N
    solver.rho[] = rho
    solver.A[] = copy(A)
    solver.B[] = copy(B)
    solver.Q[] = copy(Q)
    solver.R[] = copy(R)
    
    status = ccall((:setup_solver, _lib_path()), Int32,
                   (Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32, Float64, Int32, Int32, Int32,
                    Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32,
                    Int32),
                   A, size(A,1), size(A,2), B, size(B,1), size(B,2), f_matrix, size(f_matrix,1), size(f_matrix,2),
                   Q, size(Q,1), size(Q,2), R, size(R,1), size(R,2), rho, nx, nu, N,
                   x_min, size(x_min,1), size(x_min,2), x_max, size(x_max,1), size(x_max,2),
                   u_min, size(u_min,1), size(u_min,2), u_max, size(u_max,1), size(u_max,2),
                   verbose ? 1 : 0)
    
    status == 0 ? (solver.is_setup[] = true) : error("Setup failed with status: $status")
    verbose && println("TinyMPC solver setup successful")
    return status
end

# Core solver functions (minimal error checking)
function set_x0(solver::TinyMPCSolver, x0::Vector{Float64}; verbose::Bool=false)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    x0_matrix = reshape(x0, length(x0), 1)
    status = ccall((:set_x0, _lib_path()), Int32, (Ptr{Float64}, Int32, Int32, Int32), 
                   x0_matrix, size(x0_matrix,1), size(x0_matrix,2), verbose ? 1 : 0)
    status != 0 && error("Failed to set initial state")
    return status
end

function set_x_ref(solver::TinyMPCSolver, x_ref::Matrix{Float64}; verbose::Bool=false)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    status = ccall((:set_x_ref, _lib_path()), Int32, (Ptr{Float64}, Int32, Int32, Int32), 
                   x_ref, size(x_ref,1), size(x_ref,2), verbose ? 1 : 0)
    status != 0 && error("Failed to set state reference")
    return status
end

function set_u_ref(solver::TinyMPCSolver, u_ref::Matrix{Float64}; verbose::Bool=false)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    status = ccall((:set_u_ref, _lib_path()), Int32, (Ptr{Float64}, Int32, Int32, Int32), 
                   u_ref, size(u_ref,1), size(u_ref,2), verbose ? 1 : 0)
    status != 0 && error("Failed to set input reference")
    return status
end

function solve(solver::TinyMPCSolver; verbose::Bool=false)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    status = ccall((:solve_mpc, _lib_path()), Int32, (Int32,), verbose ? 1 : 0)
    status != 0 && error("Solver failed")
    return status
end

function get_solution(solver::TinyMPCSolver)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    
    nx, nu, N = solver.nx[], solver.nu[], solver.N[]
    
    # Allocate buffers
    states_buffer = zeros(Float64, nx * N)
    controls_buffer = zeros(Float64, nu * (N-1))
    
    # Get dimensions
    states_rows, states_cols = Ref{Int32}(), Ref{Int32}()
    controls_rows, controls_cols = Ref{Int32}(), Ref{Int32}()
    
    # Get states and controls
    status_states = ccall((:get_states, _lib_path()), Int32, (Ptr{Float64}, Ref{Int32}, Ref{Int32}),
                         states_buffer, states_rows, states_cols)
    status_controls = ccall((:get_controls, _lib_path()), Int32, (Ptr{Float64}, Ref{Int32}, Ref{Int32}),
                           controls_buffer, controls_rows, controls_cols)
    
    (status_states != 0 || status_controls != 0) && error("Failed to get solution")
    
    # Reshape to matrices (convert Int32 to Int)
    states = reshape(states_buffer[1:Int(states_rows[])*Int(states_cols[])], Int(states_rows[]), Int(states_cols[]))
    controls = reshape(controls_buffer[1:Int(controls_rows[])*Int(controls_cols[])], Int(controls_rows[]), Int(controls_cols[]))
    
    return (states=states, controls=controls)
end

function set_bound_constraints(solver::TinyMPCSolver, 
                              x_min::Matrix{Float64}, x_max::Matrix{Float64},
                              u_min::Matrix{Float64}, u_max::Matrix{Float64}; verbose::Bool=false)
    _ensure_loaded()
    status = ccall((:set_bound_constraints, _lib_path()), Int32,
                   (Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32, Int32),
                   x_min, size(x_min,1), size(x_min,2), x_max, size(x_max,1), size(x_max,2),
                   u_min, size(u_min,1), size(u_min,2), u_max, size(u_max,1), size(u_max,2),
                   verbose ? 1 : 0)
    status != 0 && error("Failed to set bound constraints")
    return status
end

function update_settings(solver::TinyMPCSolver; kwargs...)
    _ensure_loaded()
    # Implementation would go here - simplified for now
    return 0
end

function print_problem_data(solver::TinyMPCSolver; verbose::Bool=false)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    ccall((:print_problem_data, _lib_path()), Int32, (Int32,), verbose ? 1 : 0)
end

"""
    compute_cache_terms(solver, A, B, Q, R; rho)

Compute LQR cache matrices for the given system.
"""
function compute_cache_terms(solver::TinyMPCSolver, A::Matrix{Float64}, B::Matrix{Float64},
                            Q::Matrix{Float64}, R::Matrix{Float64}; rho::Float64=1.0)
    # Solve LQR problem and return cache matrices
    K, P, C1, C2 = solve_lqr(A, B, Q, R, rho)
    return (Kinf=K, Pinf=P, Quu_inv=C1, AmBKt=C2)
end

function set_cache_terms(solver::TinyMPCSolver, Kinf::Matrix{Float64}, Pinf::Matrix{Float64},
                        Quu_inv::Matrix{Float64}, AmBKt::Matrix{Float64}; verbose::Bool=false)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    
    status = ccall((:set_cache_terms, _lib_path()), Int32,
                   (Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32, Int32),
                   Kinf, size(Kinf,1), size(Kinf,2), Pinf, size(Pinf,1), size(Pinf,2),
                   Quu_inv, size(Quu_inv,1), size(Quu_inv,2), AmBKt, size(AmBKt,1), size(AmBKt,2),
                   verbose ? 1 : 0)
    
    status != 0 && error("Failed to set cache terms")
    return status
end

function set_sensitivity_matrices(solver::TinyMPCSolver, dK::Matrix{Float64}, dP::Matrix{Float64},
                                 dC1::Matrix{Float64}, dC2::Matrix{Float64}; rho::Union{Float64,Nothing}=nothing, verbose::Bool=false)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    
    # Update rho if provided
    if rho !== nothing
        solver.rho[] = rho
    end
    
    status = ccall((:set_sensitivity_matrices, _lib_path()), Int32,
                   (Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32, Int32),
                   dK, size(dK,1), size(dK,2), dP, size(dP,1), size(dP,2),
                   dC1, size(dC1,1), size(dC1,2), dC2, size(dC2,1), size(dC2,2),
                   verbose ? 1 : 0)
    
    status != 0 && error("Failed to set sensitivity matrices")
    return status
end

"""
    compute_sensitivity_autograd(solver)

Compute sensitivity matrices using numerical differentiation.
This mimics Python's Autograd behavior but uses finite differences.
NOTE: THIS IS NUMERICAL DIFFERENTIATION, NOT SYMBOLIC DIFFERENTIATION
"""
function compute_sensitivity_autograd(solver::TinyMPCSolver)
    solver.is_setup[] || error("Solver not setup")
    
    println("Computing sensitivity matrices using numerical differentiation")
    
    # Use finite differences - robust and fast for all system sizes
    h = 1e-6  # Step size for numerical differentiation
    A, B, Q, R, rho = solver.A[], solver.B[], solver.Q[], solver.R[], solver.rho[]
    
    # Compute LQR matrices at current rho
    K0, P0, C1_0, C2_0 = solve_lqr(A, B, Q, R, rho)
    
    # Compute LQR matrices at rho + h
    K1, P1, C1_1, C2_1 = solve_lqr(A, B, Q, R, rho + h)
    
    # Compute derivatives using finite differences: d/drho ≈ (f(rho+h) - f(rho)) / h
    dK = (K1 - K0) / h
    dP = (P1 - P0) / h
    dC1 = (C1_1 - C1_0) / h
    dC2 = (C2_1 - C2_0) / h
    
    println("Sensitivity matrices computed successfully")
    
    return (dK, dP, dC1, dC2)
end

# Legacy API compatibility for tests
function compute_sensitivity_autograd(solver::TinyMPCSolver, A::Matrix{Float64}, B::Matrix{Float64},
                                     Q::Matrix{Float64}, R::Matrix{Float64}, rho::Float64; verbose::Bool=false)
    verbose && println("Computing sensitivity matrices using numerical differentiation (legacy API)")
    
    # Use finite differences - robust and fast for all system sizes
    h = 1e-6  # Step size for numerical differentiation
    
    # Compute LQR matrices at current rho
    K0, P0, C1_0, C2_0 = solve_lqr(A, B, Q, R, rho)
    
    # Compute LQR matrices at rho + h
    K1, P1, C1_1, C2_1 = solve_lqr(A, B, Q, R, rho + h)
    
    # Compute derivatives using finite differences: d/drho ≈ (f(rho+h) - f(rho)) / h
    dK = (K1 - K0) / h
    dP = (P1 - P0) / h
    dC1 = (C1_1 - C1_0) / h
    dC2 = (C2_1 - C2_0) / h
    
    verbose && println("Sensitivity matrices computed successfully")
    
    # Return as NamedTuple to match test expectations
    return (dK=dK, dP=dP, dC1=dC1, dC2=dC2)
end

# Private helper function for LQR solving
function solve_lqr(A::Matrix{Float64}, B::Matrix{Float64}, Q::Matrix{Float64}, R::Matrix{Float64}, rho_val::Float64)
    # Solve LQR problem for given rho value and return all matrices
    nx, nu = size(A, 1), size(B, 2)
    
    # Add rho regularization to cost matrices
    Q_rho = Q + rho_val * I(nx)
    R_rho = R + rho_val * I(nu)
    
    # Solve discrete-time algebraic Riccati equation iteratively
    P = copy(Q_rho)
    K = zeros(nu, nx)
    
    for iter in 1:5000
        K_prev = copy(K)
        K = (R_rho + B' * P * B + 1e-8*I(nu)) \ (B' * P * A)
        P = Q_rho + A' * P * (A - B * K)
        
        iter > 1 && norm(K - K_prev) < 1e-10 && break
    end
    
    # Compute cache matrices that TinyMPC needs
    C1 = inv(R_rho + B' * P * B)  # Quu_inv matrix
    C2 = (A - B * K)'             # AmBKt matrix
    
    return K, P, C1, C2
end

"""
    codegen(solver, output_dir; verbose=false)

Generate standalone C++ code.
"""
function codegen(solver::TinyMPCSolver, output_dir::String; verbose::Bool=false)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    
    status = ccall((:codegen, _lib_path()), Int32, (Cstring, Int32), output_dir, verbose ? 1 : 0)
    status != 0 && error("Code generation failed")
    verbose && println("Code generation completed successfully in $output_dir")
    return status
end

"""
    codegen_with_sensitivity(solver, output_dir, dK, dP, dC1, dC2; verbose=false)

Generate standalone C++ code with sensitivity matrices for adaptive rho.
"""
function codegen_with_sensitivity(solver::TinyMPCSolver, output_dir::String,
                                 dK::Matrix{Float64}, dP::Matrix{Float64},
                                 dC1::Matrix{Float64}, dC2::Matrix{Float64}; verbose::Bool=false)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    
    # Set sensitivity matrices first
    set_sensitivity_matrices(solver, dK, dP, dC1, dC2; verbose=verbose)
    
    status = ccall((:codegen_with_sensitivity, _lib_path()), Int32,
                   (Cstring, Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32, Int32),
                   output_dir, dK, size(dK,1), size(dK,2), dP, size(dP,1), size(dP,2),
                   dC1, size(dC1,1), size(dC1,2), dC2, size(dC2,1), size(dC2,2), verbose ? 1 : 0)
    
    status != 0 && error("Code generation with sensitivity failed")
    verbose && println("Code generation with sensitivity completed successfully in $output_dir")
    return status
end

# Cleanup function
cleanup() = try; ccall((:cleanup_solver, _lib_path()), Cvoid, ()); catch; end
atexit(cleanup)

end # module 