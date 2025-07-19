# TinyMPC Julia Interface

Julia wrapper for [TinyMPC](https://tinympc.org/). Supports code generation and interaction with the C/C++ backend. Tested on Ubuntu and macOS.


## Prerequisites

- Julia 1.6 or later
- C++ compiler with C++17 support  
- No additional packages required (numerical differentiation used instead of ForwardDiff.jl)


## Building

1. **Clone this repo (with submodules):**
   ```bash
   git clone --recurse-submodules https://github.com/TinyMPC/tinympc-julia.git
   ```
   If you already cloned without `--recurse-submodules`, run:
   ```bash
   git submodule update --init --recursive
   ```

2. **Install and build:**
   ```bash
   cd tinympc-julia
   
   # Develop the package in Julia
   julia -e "using Pkg; Pkg.develop(PackageSpec(path=\".\"))"

   # Build the C++ library (automatically runs deps/build.jl)
   julia -e "using Pkg; Pkg.build(\"TinyMPC\")"
   ```

3. **Verify installation:**
   ```bash
   # Test that the module loads correctly
   julia -e "using TinyMPC; solver = TinyMPCSolver(); println(\"✅ TinyMPC.jl ready to use!\")"
   ```

## Examples

The `examples/` directory contains scripts demonstrating TinyMPC features:
- `cartpole_one_solve_demo.jl` - One-step solve
- `cartpole_example_mpc.jl` - Full MPC loop  
- `cartpole_example_reference_constrained.jl` - Reference tracking and constraints
- `cartpole_example_code_generation.jl` - Code generation
- `quadrotor_hover_codegen.jl` - Quadrotor codegen

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

# Set initial state and solve
x0 = [0.5; 0; 0; 0]  # Initial state
set_x0(solver, x0)
solution = solve(solver)

# Access solution
println("First control: $(solution.controls[1])")
states_trajectory = solution.states_all      # All predicted states  
controls_trajectory = solution.controls_all  # All predicted controls
```

### Code Generation Workflow

```julia
# Setup solver with constraints
solver = TinyMPCSolver()
u_min = fill(-0.5, 1, 1); u_max = fill(0.5, 1, 1)  # Control bounds
setup(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, u_min=u_min, u_max=u_max)

# Generate C++ code
codegen(solver, "out")
```

### Adaptive Rho Workflow

```julia
# Compute sensitivity matrices using built-in numerical differentiation
dK, dP, dC1, dC2 = compute_sensitivity_autograd(solver)

# Generate code with sensitivity matrices
codegen_with_sensitivity(solver, "out", dK, dP, dC1, dC2)
```

See `examples/quadrotor_hover_codegen.jl` for a complete example.

## Key Features

The test suite verifies all core functionality:

## Notes on Sensitivity Computation

- The method `compute_sensitivity_autograd()` uses numerical finite differences to compute derivatives of the LQR solution with respect to `rho` (just like MATLAB)
- This is faster and more reliable than automatic differentiation, especially for larger systems
- Uses step size `h = 1e-6` for numerical differentiation: `d/drho ≈ (f(rho+h) - f(rho)) / h`
- Required for adaptive rho workflows and generating code with sensitivity matrices

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
solution = solve(solver)
states = get_states(solver)
controls = get_controls(solver)
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

The `solve()` function returns a solution object with:
- `solution.controls` - Optimal control sequence
- `solution.states_all` - Full state trajectory 
- `solution.controls_all` - Full control trajectory
- `solution.iter` - Number of iterations
- `solution.solved` - Success flag

## Testing

Run the test suite:
```bash
julia -e "using Pkg; Pkg.test(\"TinyMPC\")"
```

Or run individual test files:
```bash  
julia tests/test_setup.jl
julia tests/test_solve.jl
julia tests/test_codegen.jl
```
