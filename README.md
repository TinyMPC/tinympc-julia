# TinyMPC Julia Interface

Julia wrapper for [TinyMPC](https://tinympc.org/). Supports code generation and interaction with the C/C++ backend. Tested on Ubuntu and macOS.

## Installation

1. **Clone this repo (with submodules):**
   ```bash
   git clone --recurse-submodules https://github.com/TinyMPC/tinympc-julia.git
   cd tinympc-julia
   ```
   If you already cloned without `--recurse-submodules`, run:
   ```bash
   git submodule update --init --recursive
   ```

2. **Install dependencies and activate the package:**
   ```bash
   # Start Julia in the tinympc-julia directory
   julia --project=.
   ```
   
   Then in the Julia REPL:
   ```julia
   # Install all dependencies (including Plots.jl)
   using Pkg
   Pkg.instantiate()
   
   
   
   # Build the C++ library
   Pkg.build("TinyMPC")
 
   # Test that everything works
   using TinyMPC
   solver = TinyMPCSolver()
   ```
   
   **Note:** You may see warnings about `Pkg.resolve()` - these are normal and can be ignored.
   
   If you see `TinyMPCSolver(Base.RefValue{Int64}(0), ...)` printed after creating the solver, installation was successful!

### Running Examples

After installation, you can run any example:
```bash
# From the tinympc-julia directory
julia --project=. examples/cartpole_example_one_solve.jl
julia --project=. examples/cartpole_example_mpc.jl
julia --project=. examples/cartpole_example_reference_constrained.jl
julia --project=. examples/cartpole_example_code_generation.jl
julia --project=. examples/quadrotor_hover_codegen.jl
julia --project=. examples/cartpole_interactive_animation.jl
```

**Note:** The `quadrotor_hover_codegen.jl` example requires ForwardDiff for automatic differentiation (already installed above) You can install it with `Pkg.add("ForwardDiff")`.

## Examples

The `examples/` directory contains scripts demonstrating TinyMPC features:
- `cartpole_example_one_solve.jl` - One-step solve
- `cartpole_example_mpc.jl` - Full MPC loop  
- `cartpole_example_reference_constrained.jl` - Reference tracking and constraints
- `cartpole_example_code_generation.jl` - Code generation
- `quadrotor_hover_codegen.jl` - Quadrotor codegen with sensitivity analysis
- `cartpole_interactive_animation.jl` - Animation from a cartpole problem

## Usage Example

### Basic MPC Workflow

```julia
using TinyMPC
using LinearAlgebra

# System matrices (cartpole example)
A = [1.0  0.01  0.0   0.0;
     0.0  1.0   0.039 0.0;
     0.0  0.0   1.002 0.01;
     0.0  0.0   0.458 1.002]
B = reshape([0.0; 0.02; 0.0; 0.067], 4, 1)
Q = diagm([10.0, 1.0, 10.0, 1.0])
R = diagm([1.0])
N = 20  # Horizon length
rho = 1.0

# Create and setup solver
solver = TinyMPCSolver()
setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)

# Set initial state and references
x0 = [0.5; 0; 0; 0]  # Initial state
set_x0(solver, x0)
set_x_ref(solver, zeros(4, N))      # State reference trajectory
set_u_ref(solver, zeros(1, N-1))    # Control reference trajectory

# Solve and get solution
status = solve(solver)  # Returns status code (0 = success)
solution = get_solution(solver)  # Get actual solution

# Access solution
println("First control: $(solution.controls[1])")
states_trajectory = solution.states      # All predicted states (4×20)
controls_trajectory = solution.controls  # All predicted controls (1×19)
```

### Code Generation Workflow

```julia
# Setup solver with constraints
solver = TinyMPCSolver()
u_min = fill(-0.5, 1, N-1); u_max = fill(0.5, 1, N-1)  # Control bounds (1×19)
setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, u_min=u_min, u_max=u_max)

# Generate C++ code
codegen(solver, "out")
```

### Adaptive Rho Workflow

```julia
# Setup solver first
solver = TinyMPCSolver()
setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N)

# Compute sensitivity matrices using built-in numerical differentiation
dK, dP, dC1, dC2 = compute_sensitivity_autograd(solver)

# Generate code with sensitivity matrices
codegen_with_sensitivity(solver, "out", dK, dP, dC1, dC2)
```

See `examples/quadrotor_hover_codegen.jl` for a complete example.

## API Reference

### Core Functions

```julia
# Setup solver with system matrices
setup(solver, A, B, fdyn, Q, R, rho, nx, nu, N; kwargs...)

# Set initial state and references 
set_x0(solver, x0)
set_x_ref(solver, x_ref)  
set_u_ref(solver, u_ref)

# Solve and get solution
status = solve(solver)        # Returns status code (Int32): 0 = success
solution = get_solution(solver)  # Returns (states=Matrix, controls=Matrix)
```

### Code Generation

```julia
# Generate standalone C++ code
codegen(solver, output_dir)

# Generate code with sensitivity matrices  
codegen_with_sensitivity(solver, output_dir, dK, dP, dC1, dC2)
```

### Sensitivity Analysis

```julia
# Compute sensitivity matrices using built-in autograd function
dK, dP, dC1, dC2 = compute_sensitivity_autograd(solver)
```

### Configuration  

```julia
# Update solver settings
update_settings(solver; abs_pri_tol=1e-6, abs_dua_tol=1e-6, max_iter=100, kwargs...)

# Set bound constraints
set_bound_constraints(solver, x_min, x_max, u_min, u_max)
```

## Solution Structure

The `get_solution()` function returns a NamedTuple with:
- `solution.states` - Full state trajectory (nx × N matrix)
- `solution.controls` - Optimal control sequence (nu × (N-1) matrix)

See [https://tinympc.org/](https://tinympc.org/) for full documentation.