module TinyMPC

export TinyMPCSolver, setup!, solve!, get_solution, set_x0!, set_x_ref!, set_u_ref!, 
       set_bound_constraints!, set_cone_constraints!, set_linear_constraints!,
       get_iterations, is_solved, update_settings!, compute_cache_terms,
       set_sensitivity_matrices!, codegen, codegen_with_sensitivity, reset!

using LinearAlgebra
using Libdl

# Define the path to the C++ library
const lib_name = "libtinympc_jl"
const lib_path = joinpath(@__DIR__, "..", "lib", lib_name)

# Global flag to track if library has been validated
const _lib_validated = Ref(false)

# Check if library exists and validate it
function _validate_library()
    if !_lib_validated[]
        full_lib_path = lib_path * "." * Libdl.dlext
        if !isfile(full_lib_path)
            error("TinyMPC library not found at $full_lib_path. Please build the package first with: julia --project=. -e \"using Pkg; Pkg.build()\"")
        end
        _lib_validated[] = true
    end
end

# Safe initialization function (precompilation-safe)
function __init__()
    # Don't perform file system operations during precompilation
    # Library validation happens on first use instead
    println("TinyMPC Julia wrapper loaded successfully")
end

"""
    TinyMPCSolver

A lightweight MPC solver instance. Use `setup!` to configure the problem
and `solve!` to compute optimal trajectories.

# Example
```julia
solver = TinyMPCSolver()
setup!(solver, A, B, f, Q, R, rho, nx, nu, N)
set_x0!(solver, x0)
solve!(solver)
solution = get_solution(solver)
```
"""
struct TinyMPCSolver
    # Store problem dimensions for later use
    nx::Ref{Int}
    nu::Ref{Int}
    N::Ref{Int}
    is_initialized::Ref{Bool}
    
    TinyMPCSolver() = new(Ref(0), Ref(0), Ref(0), Ref(false))
end

"""
    setup!(solver, A, B, f, Q, R, rho, nx, nu, N; verbose=false)

Setup the MPC problem with system matrices and parameters.

# Arguments
- `solver`: TinyMPCSolver instance  
- `A`: State transition matrix (nx × nx)
- `B`: Input matrix (nx × nu)
- `f`: Affine dynamics term (nx × 1)
- `Q`: State cost matrix (nx × nx)
- `R`: Input cost matrix (nu × nu)  
- `rho`: ADMM penalty parameter
- `nx`: Number of states
- `nu`: Number of inputs
- `N`: Prediction horizon

# Returns
Status code (0 for success)
"""
function setup!(solver::TinyMPCSolver, A::Matrix{Float64}, B::Matrix{Float64}, f::Vector{Float64},
                Q::Matrix{Float64}, R::Matrix{Float64}, rho::Float64, 
                nx::Int, nu::Int, N::Int; verbose::Bool=false)
    
    # Validate library on first use
    _validate_library()
    
    # Create default bound constraints (no constraints)
    x_min = fill(-1e17, nx, N)
    x_max = fill(1e17, nx, N)
    u_min = fill(-1e17, nu, N-1)
    u_max = fill(1e17, nu, N-1)
    
    # Reshape f to be a matrix for consistency with C++ interface
    f_matrix = reshape(f, nx, 1)
    
    # Store problem dimensions
    solver.nx[] = nx
    solver.nu[] = nu
    solver.N[] = N
    
    # Call C++ setup function with simple C interface
    status = ccall((:setup_solver, lib_path), Int32,
                   (Ptr{Float64}, Int32, Int32,     # A
                    Ptr{Float64}, Int32, Int32,     # B
                    Ptr{Float64}, Int32, Int32,     # fdyn
                    Ptr{Float64}, Int32, Int32,     # Q
                    Ptr{Float64}, Int32, Int32,     # R
                    Float64, Int32, Int32, Int32,   # rho, nx, nu, N
                    Ptr{Float64}, Int32, Int32,     # x_min
                    Ptr{Float64}, Int32, Int32,     # x_max
                    Ptr{Float64}, Int32, Int32,     # u_min
                    Ptr{Float64}, Int32, Int32,     # u_max
                    Int32),                         # verbose
                   A, size(A, 1), size(A, 2),
                   B, size(B, 1), size(B, 2),
                   f_matrix, size(f_matrix, 1), size(f_matrix, 2),
                   Q, size(Q, 1), size(Q, 2),
                   R, size(R, 1), size(R, 2),
                   rho, nx, nu, N,
                   x_min, size(x_min, 1), size(x_min, 2),
                   x_max, size(x_max, 1), size(x_max, 2),
                   u_min, size(u_min, 1), size(u_min, 2),
                   u_max, size(u_max, 1), size(u_max, 2),
                   verbose ? 1 : 0)
    
    if status == 0
        solver.is_initialized[] = true
        if verbose
            println("TinyMPC solver setup successful (nx=$nx, nu=$nu, N=$N)")
        end
    else
        error("TinyMPC solver setup failed with status: $status")
    end
    
    return status
end

"""
    set_x0!(solver, x0)

Set the initial state for the MPC problem.

# Arguments
- `solver`: TinyMPCSolver instance
- `x0`: Initial state vector (nx × 1)
"""
function set_x0!(solver::TinyMPCSolver, x0::Vector{Float64}; verbose::Bool=false)
    status = check_initialized(solver)
    if status != 0
        return status
    end
    _validate_library()
    
    # Reshape to matrix for C++ interface
    x0_matrix = reshape(x0, length(x0), 1)
    
    status = ccall((:set_x0, lib_path), Int32,
                   (Ptr{Float64}, Int32, Int32, Int32),
                   x0_matrix, size(x0_matrix, 1), size(x0_matrix, 2), verbose ? 1 : 0)
    
    if status != 0
        error("Failed to set initial state with status: $status")
    end
    
    return status
end

"""
    set_x_ref!(solver, x_ref)

Set state reference trajectory.

# Arguments  
- `solver`: TinyMPCSolver instance
- `x_ref`: State reference trajectory (nx × N)
"""
function set_x_ref!(solver::TinyMPCSolver, x_ref::Matrix{Float64}; verbose::Bool=false)
    status = check_initialized(solver)
    if status != 0
        return status
    end
    _validate_library()
    
    status = ccall((:set_x_ref, lib_path), Int32,
                   (Ptr{Float64}, Int32, Int32, Int32),
                   x_ref, size(x_ref, 1), size(x_ref, 2), verbose ? 1 : 0)
    
    if status != 0
        error("Failed to set state reference with status: $status")
    end
    
    return status
end

"""
    set_u_ref!(solver, u_ref)

Set input reference trajectory.

# Arguments
- `solver`: TinyMPCSolver instance  
- `u_ref`: Input reference trajectory (nu × N-1)
"""
function set_u_ref!(solver::TinyMPCSolver, u_ref::Matrix{Float64}; verbose::Bool=false)
    status = check_initialized(solver)
    if status != 0
        return status
    end
    _validate_library()
    
    status = ccall((:set_u_ref, lib_path), Int32,
                   (Ptr{Float64}, Int32, Int32, Int32),
                   u_ref, size(u_ref, 1), size(u_ref, 2), verbose ? 1 : 0)
    
    if status != 0
        error("Failed to set input reference with status: $status")
    end
    
    return status
end

"""
    set_bound_constraints!(solver, x_min, x_max, u_min, u_max)

Set box constraints on states and inputs.

# Arguments
- `solver`: TinyMPCSolver instance
- `x_min`: State lower bounds (nx × N)
- `x_max`: State upper bounds (nx × N)  
- `u_min`: Input lower bounds (nu × N-1)
- `u_max`: Input upper bounds (nu × N-1)
"""
function set_bound_constraints!(solver::TinyMPCSolver, 
                               x_min::Matrix{Float64}, x_max::Matrix{Float64},
                               u_min::Matrix{Float64}, u_max::Matrix{Float64};
                               verbose::Bool=false)
    status = ccall((:set_bound_constraints, lib_path), Int32,
                   (Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32,
                    Int32),
                   x_min, size(x_min,1), size(x_min,2),
                   x_max, size(x_max,1), size(x_max,2),
                   u_min, size(u_min,1), size(u_min,2),
                   u_max, size(u_max,1), size(u_max,2),
                   verbose ? 1 : 0)
    if status != 0
        error("Failed to set bound constraints with status: $status")
    end

    return status
end

"""
    solve!(solver; verbose=false)

Solve the MPC optimization problem.

# Arguments
- `solver`: TinyMPCSolver instance

# Returns
Status code (0 for success)
"""
function solve!(solver::TinyMPCSolver; verbose::Bool=false)
    status = check_initialized(solver)
    if status != 0
        return status
    end
    _validate_library()
    
    status = ccall((:solve_mpc, lib_path), Int32, (Int32,), verbose ? 1 : 0)
    
    if verbose && status != 0
        println("Solver finished with status: $status")
    end
    
    return status
end

"""
    get_solution(solver)

Get the optimal state and control trajectories.

# Arguments
- `solver`: TinyMPCSolver instance

# Returns
Named tuple with fields:
- `states`: Optimal state trajectory (nx × N)  
- `controls`: Optimal control trajectory (nu × N-1)
"""
function get_solution(solver::TinyMPCSolver)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end
    _validate_library()
    
    # Get solution dimensions first
    nx, nu, N = solver.nx[], solver.nu[], solver.N[]
    
    # Allocate buffers for states and controls
    states_buffer = zeros(Float64, nx * N)
    controls_buffer = zeros(Float64, nu * (N-1))
    
    # Get dimensions from C++
    states_rows = Ref{Int32}()
    states_cols = Ref{Int32}()
    controls_rows = Ref{Int32}()
    controls_cols = Ref{Int32}()
    
    # Get states
    status_states = ccall((:get_states, lib_path), Int32,
                         (Ptr{Float64}, Ref{Int32}, Ref{Int32}),
                         states_buffer, states_rows, states_cols)
    
    if status_states != 0
        error("Failed to get solution states")
    end
    
    # Get controls
    status_controls = ccall((:get_controls, lib_path), Int32,
                           (Ptr{Float64}, Ref{Int32}, Ref{Int32}),
                           controls_buffer, controls_rows, controls_cols)
    
    if status_controls != 0
        error("Failed to get solution controls")
    end
    
    # Reshape buffers to matrices (Julia is column-major like Eigen)
    states = reshape(states_buffer[1:(states_rows[] * states_cols[])], Int(states_rows[]), Int(states_cols[]))
    controls = reshape(controls_buffer[1:(controls_rows[] * controls_cols[])], Int(controls_rows[]), Int(controls_cols[]))
    
    return (states=states, controls=controls)
end

"""
    get_iterations(solver)

Get the number of iterations from the last solve.

# Arguments
- `solver`: TinyMPCSolver instance

# Returns
Number of iterations (-1 if not available)
"""
function get_iterations(solver::TinyMPCSolver)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end
    _validate_library()
    
    iterations = ccall((:get_iterations, lib_path), Int32, ())
    return Int(iterations)
end

"""
    is_solved(solver)

Check if the problem was successfully solved.

# Arguments  
- `solver`: TinyMPCSolver instance

# Returns
True if solved, false otherwise
"""
function is_solved(solver::TinyMPCSolver)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end
    _validate_library()
    
    solved = ccall((:is_solved, lib_path), Int32, ())
    return solved != 0
end

# (Removed placeholder update_settings! here - see advanced implementation below)

"""
    codegen(solver, output_dir; verbose=false)

Generate standalone C++ code for the configured MPC problem.

# Arguments
- `solver`: TinyMPCSolver instance
- `output_dir`: Directory to write generated code
- `verbose`: Print generation details
"""
function codegen(solver::TinyMPCSolver, output_dir::String; verbose::Bool=false)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end
    _validate_library()
    
    status = ccall((:codegen, lib_path), Int32,
                   (Cstring, Int32),
                   output_dir, verbose ? 1 : 0)
    
    if status == 0
        _copy_build_artifacts(output_dir)
        if verbose
            println("Code generation completed successfully in $output_dir")
        end
    else
        error("Code generation failed with status: $status")
    end
    
    return status
end

function _copy_build_artifacts(output_dir::String)
    try
        codegen_src_path = joinpath(@__DIR__, "codegen_src")
        if isdir(codegen_src_path)
            println("Copying codegen_src contents to output directory…")
            for (root, dirs, files) in walkdir(codegen_src_path)
                # Determine relative path w.r.t. codegen_src_path
                rel_parts = splitpath(root)[length(splitpath(codegen_src_path))+1:end]
                dest_dir = isempty(rel_parts) ? output_dir : joinpath(output_dir, joinpath(rel_parts...))
                mkpath(dest_dir)
                for f in files
                    cp(joinpath(root, f), joinpath(dest_dir, f); force=true)
                end
            end
        else
            @warn "codegen_src directory not found at $codegen_src_path"
        end

        build_path = joinpath(output_dir, "build")
        if !isdir(build_path)
            mkpath(build_path)
        end
    catch err
        @warn "Error copying build artifacts: $(err)"
    end
end

# ------------------ Advanced Constraint APIs ------------------
"""
    set_cone_constraints!(solver, Acu, qcu, cu, Acx, qcx, cx; verbose=false)

Set second-order cone (SOC) constraints on states and inputs.
The arguments follow the MATLAB/Python convention:
- `Acu`: Vector of start indices for each input cone (Vector{Int})
- `qcu`: Vector of dimensions for each input cone (Vector{Int})
- `cu`:  Vector of mu coefficients for each input cone (Vector{Float64})
- `Acx`, `qcx`, `cx`: Same but for state cones
"""
function set_cone_constraints!(solver::TinyMPCSolver,
                               Acu::Vector{Int}, qcu::Vector{Int}, cu::Vector{Float64},
                               Acx::Vector{Int}, qcx::Vector{Int}, cx::Vector{Float64};
                               verbose::Bool=false)
    status = check_initialized(solver)
    if status != 0
        return status
    end
    _validate_library()

    # Convert to Int32 for C interface
    Acu_i32 = Int32.(Acu)
    qcu_i32 = Int32.(qcu)
    Acx_i32 = Int32.(Acx)
    qcx_i32 = Int32.(qcx)

    status = ccall((:set_cone_constraints, lib_path), Int32,
                   (Ptr{Int32}, Int32,             # Acu
                    Ptr{Int32}, Int32,             # qcu
                    Ptr{Float64}, Int32,           # cu
                    Ptr{Int32}, Int32,             # Acx
                    Ptr{Int32}, Int32,             # qcx
                    Ptr{Float64}, Int32,           # cx
                    Int32),                        # verbose
                   Acu_i32, length(Acu_i32),
                   qcu_i32, length(qcu_i32),
                   cu, length(cu),
                   Acx_i32, length(Acx_i32),
                   qcx_i32, length(qcx_i32),
                   cx, length(cx),
                   verbose ? 1 : 0)

    if status != 0
        error("Failed to set cone constraints with status: $status")
    end
    return status
end

"""
    set_linear_constraints!(solver, Alin_x, blin_x, Alin_u, blin_u; verbose=false)

Set linear constraints of the form `Alin_x * x ≤ blin_x` and `Alin_u * u ≤ blin_u`.
"""
function set_linear_constraints!(solver::TinyMPCSolver,
                                 Alin_x::Matrix{Float64}, blin_x::Vector{Float64},
                                 Alin_u::Matrix{Float64}, blin_u::Vector{Float64};
                                 verbose::Bool=false)
    status = check_initialized(solver)
    if status != 0
        return status
    end
    _validate_library()

    status = ccall((:set_linear_constraints, lib_path), Int32,
                   (Ptr{Float64}, Int32, Int32,   # Alin_x
                    Ptr{Float64}, Int32,         # blin_x
                    Ptr{Float64}, Int32, Int32,  # Alin_u
                    Ptr{Float64}, Int32,         # blin_u
                    Int32),                      # verbose
                   Alin_x, size(Alin_x,1), size(Alin_x,2),
                   blin_x, length(blin_x),
                   Alin_u, size(Alin_u,1), size(Alin_u,2),
                   blin_u, length(blin_u),
                   verbose ? 1 : 0)
    if status != 0
        error("Failed to set linear constraints with status: $status")
    end
    return status
end

# ------------------ Settings Update ------------------
"""
    update_settings!(solver; kwargs...)

Update solver settings at runtime (abs_pri_tol, abs_dua_tol, max_iter, etc.).
Supported keyword arguments mirror the TinySettings struct.
"""
function update_settings!(solver::TinyMPCSolver; kwargs...)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end
    _validate_library()

    # Extract settings with defaults (use nothing if not provided)
    abs_pri_tol       = Float64(get(kwargs, :abs_pri_tol, 0.0))
    abs_dua_tol       = Float64(get(kwargs, :abs_dua_tol, 0.0))
    max_iter          = Int(get(kwargs, :max_iter, 0))
    check_termination = Int(get(kwargs, :check_termination, 5))
    en_state_bound    = Int(get(kwargs, :en_state_bound, 0))
    en_input_bound    = Int(get(kwargs, :en_input_bound, 0))
    en_state_soc      = Int(get(kwargs, :en_state_soc, 0))
    en_input_soc      = Int(get(kwargs, :en_input_soc, 0))
    en_state_linear   = Int(get(kwargs, :en_state_linear, 0))
    en_input_linear   = Int(get(kwargs, :en_input_linear, 0))

    status = ccall((:update_settings, lib_path), Int32,
                   (Float64, Float64, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32),
                   abs_pri_tol, abs_dua_tol, max_iter, check_termination,
                   en_state_bound, en_input_bound,
                   en_state_soc, en_input_soc,
                   en_state_linear, en_input_linear,
                   0)  # verbose disabled for now
    return status
end

# ------------------ Cache & Sensitivity ------------------
const _sensitivity_store = Dict{Symbol,Any}()

"""
    compute_cache_terms(solver, A, B, Q, R; rho=1.0)

Compute LQR-based cache terms (Kinf, Pinf, Quu_inv, AmBKt) similar to MATLAB helper.
"""
function compute_cache_terms(solver::TinyMPCSolver,
                             A::Matrix{Float64}, B::Matrix{Float64},
                             Q::Matrix{Float64}, R::Matrix{Float64};
                             rho::Float64=1.0)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end

    nx, nu = solver.nx[], solver.nu[]
    Q_rho = Q + rho * I(nx)
    R_rho = R + rho * I(nu)

    Kinf = zeros(nu, nx)
    Pinf = copy(Q)

    for _ in 1:5000
        Kprev = copy(Kinf)
        Kinf = (R_rho + B' * Pinf * B + 1e-8I(nu)) \ (B' * Pinf * A)
        Pinf = Q_rho + A' * Pinf * (A - B * Kinf)
        if norm(Kinf - Kprev) < 1e-10
            break
        end
    end

    AmBKt = (A - B * Kinf)'
    Quu_inv = inv(R_rho + B' * Pinf * B)

    return (Kinf=Kinf, Pinf=Pinf, Quu_inv=Quu_inv, AmBKt=AmBKt)
end

"""
    set_sensitivity_matrices!(solver, dK, dP, dC1, dC2)

Store sensitivity matrices for later code generation. Currently only stored in Julia layer.
"""
function set_sensitivity_matrices!(solver::TinyMPCSolver,
                                   dK::Matrix{Float64}, dP::Matrix{Float64},
                                   dC1::Matrix{Float64}, dC2::Matrix{Float64})
    _sensitivity_store[:dK] = dK
    _sensitivity_store[:dP] = dP
    _sensitivity_store[:dC1] = dC1
    _sensitivity_store[:dC2] = dC2
    return 0
end

"""
    codegen_with_sensitivity(solver, output_dir; verbose=false)

Generate C++ code including previously provided sensitivity matrices.
"""
function codegen_with_sensitivity(solver::TinyMPCSolver, output_dir::String; verbose::Bool=false)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end
    _validate_library()

    dK  = _sensitivity_store[:dK]
    dP  = _sensitivity_store[:dP]
    dC1 = _sensitivity_store[:dC1]
    dC2 = _sensitivity_store[:dC2]
    if any(x->x===nothing, (dK,dP,dC1,dC2))
        error("Sensitivity matrices not set. Call set_sensitivity_matrices! first.")
    end

    status = ccall((:codegen_with_sensitivity, lib_path), Int32,
                   (Cstring,
                    Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32,
                    Int32),
                   output_dir,
                   dK, size(dK,1), size(dK,2),
                   dP, size(dP,1), size(dP,2),
                   dC1, size(dC1,1), size(dC1,2),
                   dC2, size(dC2,1), size(dC2,2),
                   verbose ? 1 : 0)

    if status != 0
        error("Code generation with sensitivity failed with status: $status")
    end

    _copy_build_artifacts(output_dir)
    
    return status
end

# ------------------ Reset ------------------
"""
    reset!(solver)

Free solver memory and mark as uninitialized.
"""
function reset!(solver::TinyMPCSolver)
    try
        ccall((:cleanup_solver, lib_path), Cvoid, ())
    catch
        # ignore errors
    end
    solver.is_initialized[] = false
    return nothing
end

# Helper function to check if solver is initialized
function check_initialized(solver::TinyMPCSolver)
    if !solver.is_initialized[]
        return -1  # Return error status instead of throwing
    end
    return 0  # Success status
end

# Cleanup function (called automatically)
function cleanup()
    try
        ccall((:cleanup_solver, lib_path), Cvoid, ())
    catch
        # Ignore cleanup errors
    end
end

# Register cleanup to be called when module is unloaded
atexit(cleanup)

end # module 