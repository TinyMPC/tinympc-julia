module TinyMPC

export TinyMPCSolver, setup, solve, get_solution, set_x0, set_x_ref, set_u_ref,
       set_bound_constraints, set_linear_constraints, set_cone_constraints, set_equality_constraints,
       update_settings, set_cache_terms, print_problem_data,
       compute_sensitivity_autograd, codegen, codegen_with_sensitivity

using LinearAlgebra, Libdl, Printf

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
               max_iter::Int=100, check_termination::Bool=true,
               adaptive_rho::Bool=false,
               adaptive_rho_min::Float64=0.1, adaptive_rho_max::Float64=10.0,
               adaptive_rho_clipping::Bool=true)
    
    _ensure_loaded()
    
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
                    Int32),
                   A, size(A,1), size(A,2), B, size(B,1), size(B,2), f_matrix, size(f_matrix,1), size(f_matrix,2),
                   Q, size(Q,1), size(Q,2), R, size(R,1), size(R,2), rho, nx, nu, N,
                   verbose ? 1 : 0)
    
    if status == 0
        solver.is_setup[] = true
        
        # Push settings to C++ layer (including adaptive_rho settings)
        update_settings(solver,
                       abs_pri_tol=abs_pri_tol,
                       abs_dua_tol=abs_dua_tol,
                       max_iter=max_iter,
                       check_termination=check_termination,
                       en_state_bound=false,
                       en_input_bound=false,
                       en_state_soc=false,
                       en_input_soc=false,
                       en_state_linear=false,
                       en_input_linear=false,
                       adaptive_rho=adaptive_rho,
                       adaptive_rho_min=adaptive_rho_min,
                       adaptive_rho_max=adaptive_rho_max,
                       adaptive_rho_enable_clipping=adaptive_rho_clipping,
                       verbose=verbose)
        
        verbose && @printf("TinyMPC solver setup successful (nx=%d, nu=%d, N=%d)\n", nx, nu, N)
    else
        error("Setup failed with status: $status")
    end
    
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

## Remove old duplicate definition (consolidated below)

function update_settings(solver::TinyMPCSolver;
                        abs_pri_tol::Float64=1e-3,
                        abs_dua_tol::Float64=1e-3,
                        max_iter::Int=100,
                        check_termination::Bool=true,
                        en_state_bound::Bool=false,
                        en_input_bound::Bool=false,
                        en_state_soc::Bool=false,
                        en_input_soc::Bool=false,
                        en_state_linear::Bool=false,
                        en_input_linear::Bool=false,
                        adaptive_rho::Bool=false,
                        adaptive_rho_min::Float64=0.1,
                        adaptive_rho_max::Float64=10.0,
                        adaptive_rho_enable_clipping::Bool=true,
                        verbose::Bool=false)
    _ensure_loaded()
    
    status = ccall((:update_settings, _lib_path()), Int32,
                   (Float64, Float64, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32,
                    Int32, Float64, Float64, Int32, Int32),
                   abs_pri_tol, abs_dua_tol, max_iter, check_termination ? 1 : 0,
                   en_state_bound ? 1 : 0, en_input_bound ? 1 : 0,
                   en_state_soc ? 1 : 0, en_input_soc ? 1 : 0,
                   en_state_linear ? 1 : 0, en_input_linear ? 1 : 0,
                   adaptive_rho ? 1 : 0, adaptive_rho_min, adaptive_rho_max,
                   adaptive_rho_enable_clipping ? 1 : 0, verbose ? 1 : 0)
    
    status != 0 && error("Failed to update settings")
    return status
end

# Constraints API
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
    # Flags auto-enabled in C++ binding
    return status
end

function set_linear_constraints(solver::TinyMPCSolver,
                                Alin_x::Matrix{Float64}, blin_x::Vector{Float64},
                                Alin_u::Matrix{Float64}, blin_u::Vector{Float64}; verbose::Bool=false)
    _ensure_loaded()
    status = ccall((:set_linear_constraints, _lib_path()), Int32,
                   (Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32,
                    Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32),
                   Alin_x, size(Alin_x,1), size(Alin_x,2), blin_x, length(blin_x),
                   Alin_u, size(Alin_u,1), size(Alin_u,2), blin_u, length(blin_u),
                   verbose ? 1 : 0)
    status != 0 && error("Failed to set linear constraints")
    # Flags auto-enabled in C++ binding
    return status
end

function set_cone_constraints(solver::TinyMPCSolver,
                              Acu::Vector{Int32}, qcu::Vector{Int32}, cu::Vector{Float64},
                              Acx::Vector{Int32}, qcx::Vector{Int32}, cx::Vector{Float64}; verbose::Bool=false)
    _ensure_loaded()
    status = ccall((:set_cone_constraints, _lib_path()), Int32,
                   (Ptr{Int32}, Int32, Ptr{Int32}, Int32, Ptr{Float64}, Int32,
                    Ptr{Int32}, Int32, Ptr{Int32}, Int32, Ptr{Float64}, Int32, Int32),
                   Acu, length(Acu), qcu, length(qcu), cu, length(cu),
                   Acx, length(Acx), qcx, length(qcx), cx, length(cx),
                   verbose ? 1 : 0)
    status != 0 && error("Failed to set cone constraints")
    # Flags auto-enabled in C++ binding
    return status
end

function set_equality_constraints(solver::TinyMPCSolver,
                                  Aeq_x::Matrix{Float64}, beq_x::Vector{Float64};
                                  Aeq_u::Matrix{Float64}=zeros(0, size(solver.B[],2)), beq_u::Vector{Float64}=zeros(Float64, 0))
    # Implement equalities via two inequalities and delegate
    Alin_x = vcat(Aeq_x, -Aeq_x)
    blin_x = vcat(beq_x, -beq_x)
    Alin_u = vcat(Aeq_u, -Aeq_u)
    blin_u = vcat(beq_u, -beq_u)
    return set_linear_constraints(solver, Alin_x, blin_x, Alin_u, blin_u)
end

function print_problem_data(solver::TinyMPCSolver; verbose::Bool=false)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    ccall((:print_problem_data, _lib_path()), Int32, (Int32,), verbose ? 1 : 0)
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



"""
    compute_sensitivity_autograd(solver)

Compute sensitivity matrices using numerical differentiation.
This mimics Python's Autograd behavior but uses finite differences.
NOTE: THIS IS NUMERICAL DIFFERENTIATION, NOT SYMBOLIC DIFFERENTIATION
"""
function compute_sensitivity_autograd(solver::TinyMPCSolver)
    solver.is_setup[] || error("Solver not setup")
    
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
    
    return (dK, dP, dC1, dC2)
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
    copy_build_artifacts(output_dir)
    @printf("Code generation completed successfully in: %s\n", output_dir)
    return status
end

"""
    codegen_with_sensitivity(solver, output_dir, dK, dP, dC1, dC2; verbose=false)

Generate standalone C++ code with sensitivity matrices for adaptive rho.
"""
function codegen_with_sensitivity(solver::TinyMPCSolver, output_dir::String,
                                 dK::Matrix{Float64}, dP::Matrix{Float64},
                                 dC1::Matrix{Float64}, dC2::Matrix{Float64}; 
                                 verbose::Bool=false)
    solver.is_setup[] || error("Solver not setup")
    _ensure_loaded()
    
    @printf("Sensitivity matrix norms: dK=%.6e, dP=%.6e, dC1=%.6e, dC2=%.6e\n", norm(dK), norm(dP), norm(dC1), norm(dC2))
    
    # Pass sensitivity matrices directly to C++ function
    # The C++ core will check if adaptive_rho is enabled and store matrices accordingly
    status = ccall((:codegen_with_sensitivity, _lib_path()), Int32,
                   (Cstring, Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32, Ptr{Float64}, Int32, Int32, Int32),
                   output_dir, dK, size(dK,1), size(dK,2), dP, size(dP,1), size(dP,2),
                   dC1, size(dC1,1), size(dC1,2), dC2, size(dC2,1), size(dC2,2), verbose ? 1 : 0)
    
    status != 0 && error("Code generation with sensitivity failed")
    copy_build_artifacts(output_dir)
    @printf("Code generation with sensitivity matrices completed successfully in: %s\n", output_dir)
    return status
end

# Copy build artifacts
function copy_build_artifacts(output_dir::String)
    try
        # Get the path to codegen_src directory (same level as TinyMPC.jl)
        module_dir = dirname(@__FILE__)
        codegen_src_path = joinpath(module_dir, "codegen_src")
        
        if isdir(codegen_src_path)
            # Copy all contents from codegen_src to output directory
            # This copies the directory contents, not the directory itself
            for item in readdir(codegen_src_path)
                src_item = joinpath(codegen_src_path, item)
                dst_item = joinpath(output_dir, item)
                if !ispath(dst_item)  # Only copy if destination doesn't exist
                    cp(src_item, dst_item; force=false, follow_symlinks=true)
                end
            end
            @printf("Copied all contents from codegen_src to output directory\n")
        end
        
        # Create build directory if it doesn't exist
        build_path = joinpath(output_dir, "build")
        if !isdir(build_path)
            mkdir(build_path)
        end
    catch e
        @warn "Error copying build artifacts: $(e)"
    end
end

# Cleanup function
cleanup() = try; ccall((:cleanup_solver, _lib_path()), Cvoid, ()); catch; end
atexit(cleanup)

end # module 