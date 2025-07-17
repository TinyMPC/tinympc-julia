module TinyMPC

export TinyMPCSolver, setup!, solve!, get_solution, set_x0!, set_x_ref!, set_u_ref!, 
       set_bound_constraints!, update_settings!, compute_cache_terms,
       set_sensitivity_matrices!, set_cache_terms!, print_problem_data,
       compute_sensitivity_autograd, codegen, codegen_with_sensitivity

using LinearAlgebra
using Libdl
using ForwardDiff

# Completely defer all library operations to runtime
const _runtime_state = Ref{Dict{Symbol, Any}}()

# Initialize runtime state on first access
function _init_runtime_state()
    if !isassigned(_runtime_state)
        _runtime_state[] = Dict{Symbol, Any}(
            :lib_loaded => false,
            :lib_name => "libtinympc_jl",
            :sensitivity_store => Dict{Symbol, Any}()
        )
    end
    return _runtime_state[]
end

# Get library path (runtime only)
function _get_lib_path()
    state = _init_runtime_state()
    lib_name = state[:lib_name]
    return joinpath(dirname(@__DIR__), "lib", lib_name)
end

# Ensure library is loaded (runtime only)
function _ensure_library_loaded()
    state = _init_runtime_state()
    if !state[:lib_loaded]
        lib_path = _get_lib_path()
        full_lib_path = lib_path * "." * Libdl.dlext
        
        if !isfile(full_lib_path)
            error("TinyMPC library not found at $full_lib_path. Please build the package first.")
        end
        
        state[:lib_loaded] = true
    end
end

# Empty init function
function __init__()
    # Absolutely nothing here to avoid precompilation issues
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
                nx::Int, nu::Int, N::Int; 
                verbose::Bool=false,
                abs_pri_tol::Float64=1e-3,
                abs_dua_tol::Float64=1e-3,
                max_iter::Int=100,
                check_termination::Int=1,
                en_state_bound::Int=1,
                en_input_bound::Int=1,
                en_state_soc::Int=0,
                en_input_soc::Int=0,
                en_state_linear::Int=0,
                en_input_linear::Int=0,
                adaptive_rho::Bool=false,
                adaptive_rho_min::Float64=0.1,
                adaptive_rho_max::Float64=10.0,
                adaptive_rho_enable_clipping::Bool=true,
                x_min::Union{Matrix{Float64}, Nothing}=nothing,
                x_max::Union{Matrix{Float64}, Nothing}=nothing,
                u_min::Union{Matrix{Float64}, Nothing}=nothing,
                u_max::Union{Matrix{Float64}, Nothing}=nothing)
    
    # Ensure library is loaded
    _ensure_library_loaded()
    
    # Handle bound constraints
    if x_min === nothing
        x_min = fill(-1e17, nx, N)
    end
    if x_max === nothing
        x_max = fill(1e17, nx, N)
    end
    if u_min === nothing
        u_min = fill(-1e17, nu, N-1)
    end
    if u_max === nothing
        u_max = fill(1e17, nu, N-1)
    end
    
    # Reshape f to be a matrix for consistency with C++ interface
    f_matrix = reshape(f, nx, 1)
    
    # Store problem dimensions
    solver.nx[] = nx
    solver.nu[] = nu
    solver.N[] = N
    
    # Call C++ setup function with simple C interface
    status = ccall((:setup_solver, _get_lib_path()), Int32,
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
        
        # Update solver settings with provided parameters
        update_settings!(solver,
                        abs_pri_tol=abs_pri_tol,
                        abs_dua_tol=abs_dua_tol,
                        max_iter=max_iter,
                        check_termination=check_termination,
                        en_state_bound=en_state_bound != 0,
                        en_input_bound=en_input_bound != 0,
                        en_state_soc=en_state_soc != 0,
                        en_input_soc=en_input_soc != 0,
                        en_state_linear=en_state_linear != 0,
                        en_input_linear=en_input_linear != 0,
                        adaptive_rho=adaptive_rho,
                        adaptive_rho_min=adaptive_rho_min,
                        adaptive_rho_max=adaptive_rho_max,
                        adaptive_rho_enable_clipping=adaptive_rho_enable_clipping,
                        verbose=false)
        
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
    _ensure_library_loaded()
    
    # Reshape to matrix for C++ interface
    x0_matrix = reshape(x0, length(x0), 1)
    
    status = ccall((:set_x0, _get_lib_path()), Int32,
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
    _ensure_library_loaded()
    
    status = ccall((:set_x_ref, _get_lib_path()), Int32,
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
    _ensure_library_loaded()
    
    status = ccall((:set_u_ref, _get_lib_path()), Int32,
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
    status = ccall((:set_bound_constraints, _get_lib_path()), Int32,
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
    _ensure_library_loaded()
    
    status = ccall((:solve_mpc, _get_lib_path()), Int32, (Int32,), verbose ? 1 : 0)
    
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
    _ensure_library_loaded()
    
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
    status_states = ccall((:get_states, _get_lib_path()), Int32,
                         (Ptr{Float64}, Ref{Int32}, Ref{Int32}),
                         states_buffer, states_rows, states_cols)
    
    if status_states != 0
        error("Failed to get solution states")
    end
    
    # Get controls
    status_controls = ccall((:get_controls, _get_lib_path()), Int32,
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
    _ensure_library_loaded()
    
    status = ccall((:codegen, _get_lib_path()), Int32,
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



# ------------------ Settings Update ------------------
"""
    update_settings!(solver; kwargs...)

Update solver settings at runtime (abs_pri_tol, abs_dua_tol, max_iter, etc.).
Supported keyword arguments match Python/MATLAB interface.

# Arguments
- `abs_pri_tol`: Solution tolerance for primal variables
- `abs_dua_tol`: Solution tolerance for dual variables  
- `max_iter`: Maximum number of iterations before returning
- `check_termination`: Number of iterations to skip before checking termination
- `en_state_bound`: Enable or disable bound constraints on state
- `en_input_bound`: Enable or disable bound constraints on input
- `adaptive_rho`: Enable adaptive rho (logical)
- `adaptive_rho_min`: Minimum rho value (positive scalar)
- `adaptive_rho_max`: Maximum rho value (positive scalar)
- `adaptive_rho_enable_clipping`: Enable rho clipping (logical)
"""
function update_settings!(solver::TinyMPCSolver; kwargs...)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end
    _ensure_library_loaded()

    # Extract settings with defaults - only update provided settings
    abs_pri_tol       = haskey(kwargs, :abs_pri_tol) ? Float64(kwargs[:abs_pri_tol]) : 0.0
    abs_dua_tol       = haskey(kwargs, :abs_dua_tol) ? Float64(kwargs[:abs_dua_tol]) : 0.0
    max_iter          = haskey(kwargs, :max_iter) ? Int32(kwargs[:max_iter]) : 0
    check_termination = haskey(kwargs, :check_termination) ? Int32(kwargs[:check_termination]) : 0
    en_state_bound    = haskey(kwargs, :en_state_bound) ? Int32(kwargs[:en_state_bound] ? 1 : 0) : 1
    en_input_bound    = haskey(kwargs, :en_input_bound) ? Int32(kwargs[:en_input_bound] ? 1 : 0) : 1
    en_state_soc      = haskey(kwargs, :en_state_soc) ? Int32(kwargs[:en_state_soc] ? 1 : 0) : 0
    en_input_soc      = haskey(kwargs, :en_input_soc) ? Int32(kwargs[:en_input_soc] ? 1 : 0) : 0
    en_state_linear   = haskey(kwargs, :en_state_linear) ? Int32(kwargs[:en_state_linear] ? 1 : 0) : 0
    en_input_linear   = haskey(kwargs, :en_input_linear) ? Int32(kwargs[:en_input_linear] ? 1 : 0) : 0
    adaptive_rho      = haskey(kwargs, :adaptive_rho) ? Int32(kwargs[:adaptive_rho] ? 1 : 0) : 0
    adaptive_rho_min  = haskey(kwargs, :adaptive_rho_min) ? Float64(kwargs[:adaptive_rho_min]) : 0.0
    adaptive_rho_max  = haskey(kwargs, :adaptive_rho_max) ? Float64(kwargs[:adaptive_rho_max]) : 0.0
    adaptive_rho_enable_clipping = haskey(kwargs, :adaptive_rho_enable_clipping) ? Int32(kwargs[:adaptive_rho_enable_clipping] ? 1 : 0) : 1
    verbose_setting   = haskey(kwargs, :verbose) ? Int32(kwargs[:verbose] ? 1 : 0) : 0

    # Call the updated C++ function with all parameters including adaptive rho
    status = ccall((:update_settings, _get_lib_path()), Int32,
                   (Float64, Float64, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Float64, Float64, Int32, Int32),
                   abs_pri_tol, abs_dua_tol, max_iter, check_termination,
                   en_state_bound, en_input_bound, en_state_soc, en_input_soc,
                   en_state_linear, en_input_linear, adaptive_rho,
                   adaptive_rho_min, adaptive_rho_max, adaptive_rho_enable_clipping, verbose_setting)
                   
    if status != 0
        error("Failed to update settings with status: $status")
    end
    
    return status
end

"""
    print_problem_data(solver)

Print detailed solver information including solution, cache, settings, and workspace data.
Matches the Python/MATLAB debug output format.
"""
function print_problem_data(solver::TinyMPCSolver)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end
    _ensure_library_loaded()

    # Call the C++ function to print debug information  
    ccall((:print_problem_data, _get_lib_path()), Cvoid, ())
    
    return nothing
end

# ------------------ Cache & Sensitivity ------------------

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
    set_sensitivity_matrices!(solver, dK, dP, dC1, dC2; verbose=false)

Set sensitivity matrices for adaptive rho behavior.
Matches Python/MATLAB set_sensitivity_matrices functionality.

# Arguments
- `solver`: TinyMPCSolver instance
- `dK`: Derivative of feedback gain w.r.t. rho (nu x nx)
- `dP`: Derivative of value function w.r.t. rho (nx x nx)
- `dC1`: Derivative of first cache matrix w.r.t. rho (nu x nu)
- `dC2`: Derivative of second cache matrix w.r.t. rho (nx x nx)
- `verbose`: Print debug information
"""
function set_sensitivity_matrices!(solver::TinyMPCSolver,
                                   dK::Matrix{Float64}, dP::Matrix{Float64},
                                   dC1::Matrix{Float64}, dC2::Matrix{Float64};
                                   verbose::Bool=false)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end
    _ensure_library_loaded()

    # Validate dimensions
    nx, nu = solver.nx[], solver.nu[]
    
    if size(dK) != (nu, nx)
        error("dK must have size ($nu, $nx), got $(size(dK))")
    end
    if size(dP) != (nx, nx)
        error("dP must have size ($nx, $nx), got $(size(dP))")
    end
    if size(dC1) != (nu, nu)
        error("dC1 must have size ($nu, $nu), got $(size(dC1))")
    end
    if size(dC2) != (nx, nx)
        error("dC2 must have size ($nx, $nx), got $(size(dC2))")
    end

    # Call C++ function to set sensitivity matrices
    status = ccall((:set_sensitivity_matrices, _get_lib_path()), Int32,
                   (Ptr{Float64}, Int32, Int32,  # dK
                    Ptr{Float64}, Int32, Int32,  # dP
                    Ptr{Float64}, Int32, Int32,  # dC1
                    Ptr{Float64}, Int32, Int32,  # dC2
                    Int32),                      # verbose
                   dK, size(dK,1), size(dK,2),
                   dP, size(dP,1), size(dP,2),
                   dC1, size(dC1,1), size(dC1,2),
                   dC2, size(dC2,1), size(dC2,2),
                   verbose ? 1 : 0)

    if status != 0
        error("Failed to set sensitivity matrices with status: $status")
    end

    if verbose
        println("Sensitivity matrices set with norms: dK=$(norm(dK):.6f), dP=$(norm(dP):.6f), dC1=$(norm(dC1):.6f), dC2=$(norm(dC2):.6f)")
    end

    # Also store in Julia for compatibility  
    state = _init_runtime_state()
    state[:sensitivity_store][:dK] = dK
    state[:sensitivity_store][:dP] = dP
    state[:sensitivity_store][:dC1] = dC1
    state[:sensitivity_store][:dC2] = dC2

    return status
end

"""
    set_cache_terms!(solver, Kinf, Pinf, Quu_inv, AmBKt; verbose=false)

Set cache terms directly in the C++ solver for manual cache control.
Matches Python/MATLAB set_cache_terms functionality.

# Arguments
- `solver`: TinyMPCSolver instance
- `Kinf`: Infinite horizon feedback gain (nu x nx)
- `Pinf`: Infinite horizon value function (nx x nx)
- `Quu_inv`: Inverse of Quu matrix (nu x nu)
- `AmBKt`: Transpose of (A - B*K) (nx x nx)
- `verbose`: Print debug information
"""
function set_cache_terms!(solver::TinyMPCSolver,
                          Kinf::Matrix{Float64}, Pinf::Matrix{Float64},
                          Quu_inv::Matrix{Float64}, AmBKt::Matrix{Float64};
                          verbose::Bool=false)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end
    _ensure_library_loaded()

    # Validate dimensions
    nx, nu = solver.nx[], solver.nu[]
    
    if size(Kinf) != (nu, nx)
        error("Kinf must have size ($nu, $nx), got $(size(Kinf))")
    end
    if size(Pinf) != (nx, nx)
        error("Pinf must have size ($nx, $nx), got $(size(Pinf))")
    end
    if size(Quu_inv) != (nu, nu)
        error("Quu_inv must have size ($nu, $nu), got $(size(Quu_inv))")
    end
    if size(AmBKt) != (nx, nx)
        error("AmBKt must have size ($nx, $nx), got $(size(AmBKt))")
    end

    # Call C++ function to set cache terms directly in solver
    status = ccall((:set_cache_terms, _get_lib_path()), Int32,
                   (Ptr{Float64}, Int32, Int32,  # Kinf
                    Ptr{Float64}, Int32, Int32,  # Pinf
                    Ptr{Float64}, Int32, Int32,  # Quu_inv
                    Ptr{Float64}, Int32, Int32,  # AmBKt
                    Int32),                      # verbose
                   Kinf, size(Kinf,1), size(Kinf,2),
                   Pinf, size(Pinf,1), size(Pinf,2),
                   Quu_inv, size(Quu_inv,1), size(Quu_inv,2),
                   AmBKt, size(AmBKt,1), size(AmBKt,2),
                   verbose ? 1 : 0)

    if status != 0
        error("Failed to set cache terms with status: $status")
    end

    if verbose
        println("Cache terms set with norms: Kinf=$(norm(Kinf):.6f), Pinf=$(norm(Pinf):.6f)")
        println("C1=$(norm(Quu_inv):.6f), C2=$(norm(AmBKt):.6f)")
    end

    # Also store in Julia for compatibility
    state = _init_runtime_state()
    state[:sensitivity_store][:Kinf] = Kinf
    state[:sensitivity_store][:Pinf] = Pinf
    state[:sensitivity_store][:Quu_inv] = Quu_inv
    state[:sensitivity_store][:AmBKt] = AmBKt

    return status
end

"""
    compute_sensitivity_autograd(solver, A, B, Q, R, rho)

Compute sensitivity matrices dK, dP, dC1, dC2 with respect to rho using ForwardDiff.jl.
Matches Python autograd and MATLAB symbolic differentiation functionality.

# Arguments
- `solver`: TinyMPCSolver instance (for dimensions)
- `A`: State transition matrix (nx x nx)
- `B`: Control matrix (nx x nu)
- `Q`: State cost matrix (nx x nx)
- `R`: Input cost matrix (nu x nu)
- `rho`: Current rho value for differentiation

# Returns
Tuple (dK, dP, dC1, dC2) of sensitivity matrices
"""
function compute_sensitivity_autograd(solver::TinyMPCSolver,
                                     A::Matrix{Float64}, B::Matrix{Float64},
                                     Q::Matrix{Float64}, R::Matrix{Float64},
                                     rho::Float64)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end

    nx, nu = solver.nx[], solver.nu[]
    
    # Define cache computation function for differentiation
    function compute_cache_matrices(ρ::T) where T
        Q_rho = Q + ρ * I(nx)
        R_rho = R + ρ * I(nu)
        
        # Initialize matrices with proper type for ForwardDiff
        Kinf = zeros(T, nu, nx)
        Pinf = convert(Matrix{T}, Q)
        
        # Solve discrete-time algebraic Riccati equation iteratively
        for _ in 1:1000
            Kprev = copy(Kinf)
            BtPinfB = B' * Pinf * B
            Quu_inv = inv(R_rho + BtPinfB + 1e-8*I(nu))
            Kinf = Quu_inv * (B' * Pinf * A)
            Pinf = Q_rho + A' * Pinf * (A - B * Kinf)
            
            if norm(Kinf - Kprev) < 1e-12
                break
            end
        end
        
        AmBKt = (A - B * Kinf)'
        Quu_inv = inv(R_rho + B' * Pinf * B)
        
        return Kinf, Pinf, Quu_inv, AmBKt
    end
    
    # Compute derivatives using ForwardDiff
    dK = ForwardDiff.derivative(ρ -> compute_cache_matrices(ρ)[1], rho)
    dP = ForwardDiff.derivative(ρ -> compute_cache_matrices(ρ)[2], rho) 
    dC1 = ForwardDiff.derivative(ρ -> compute_cache_matrices(ρ)[3], rho)
    dC2 = ForwardDiff.derivative(ρ -> compute_cache_matrices(ρ)[4], rho)
    
    return (dK, dP, dC1, dC2)
end

"""
    codegen_with_sensitivity(solver, output_dir, dK, dP, dC1, dC2; verbose=false)

Generate standalone C++ code with sensitivity matrices for adaptive rho.
Matches Python/MATLAB codegen_with_sensitivity functionality.

# Arguments
- `solver`: TinyMPCSolver instance
- `output_dir`: Directory to write generated code
- `dK`: Derivative of feedback gain w.r.t. rho (nu x nx)
- `dP`: Derivative of value function w.r.t. rho (nx x nx)
- `dC1`: Derivative of first cache matrix w.r.t. rho (nu x nu)
- `dC2`: Derivative of second cache matrix w.r.t. rho (nx x nx)
- `verbose`: Print generation details
"""
function codegen_with_sensitivity(solver::TinyMPCSolver,
                                  output_dir::String,
                                  dK::Matrix{Float64}, dP::Matrix{Float64},
                                  dC1::Matrix{Float64}, dC2::Matrix{Float64};
                                  verbose::Bool=false)
    status = check_initialized(solver)
    if status != 0
        error("Solver not initialized. Call setup!() first.")
    end
    _ensure_library_loaded()

    # Set sensitivity matrices in the solver first
    set_sensitivity_matrices!(solver, dK, dP, dC1, dC2; verbose=verbose)

    # Call C++ function for code generation with sensitivity
    status = ccall((:codegen_with_sensitivity, _get_lib_path()), Int32,
                   (Cstring, Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32,
                    Ptr{Float64}, Int32, Int32,
                    Int32),
                   output_dir,
                   dK, size(dK, 1), size(dK, 2),
                   dP, size(dP, 1), size(dP, 2),
                   dC1, size(dC1, 1), size(dC1, 2),
                   dC2, size(dC2, 1), size(dC2, 2),
                   verbose ? 1 : 0)

    if status == 0
        _copy_build_artifacts(output_dir)
        if verbose
            println("Code generation with sensitivity completed successfully in $output_dir")
        end
    else
        error("Code generation with sensitivity failed with status: $status")
    end

    return status
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
        ccall((:cleanup_solver, _get_lib_path()), Cvoid, ())
    catch
        # Ignore cleanup errors
    end
end

# Register cleanup to be called when module is unloaded
atexit(cleanup)

end # module 