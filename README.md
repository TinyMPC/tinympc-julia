# TinyMPC Julia Interface

Julia wrapper for [TinyMPC](https://tinympc.org/). Supports code generation and interaction with the C/C++ backend. Tested on Ubuntu and macOS.

## Building

1. **Clone with submodules:**
```bash
git clone --recurse-submodules https://github.com/TinyMPC/tinympc-julia.git
cd tinympc-julia
```

2. **Install and build:**
```bash
# Develop the package in Julia
julia -e "using Pkg; Pkg.develop(PackageSpec(path=\".\"))"

# Build the C++ library (automatically runs deps/build.jl)
julia -e "using Pkg; Pkg.build()"

# Or build manually if needed:
# julia deps/build.jl
```

3. **Verify installation:**
```bash
# Test that the module loads correctly
julia -e "using TinyMPC; solver = TinyMPCSolver(); println(\"✅ TinyMPC.jl ready to use!\")"
```

## Basic Usage

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
status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N, verbose=false)

# Set initial state and references
set_x0!(solver, [0.5, 0.0, 0.0, 0.0])
set_x_ref!(solver, zeros(4, N))
set_u_ref!(solver, zeros(1, N-1))

# Solve and get solution
status = solve!(solver)
solution = get_solution(solver)

println("Optimal states: ", solution.states)
println("Optimal controls: ", solution.controls)
```

## Running Examples

All examples work out-of-the-box using the direct include approach:

```bash
# Basic one-solve example
julia --project=. --compile=min --startup-file=no examples/cartpole_one_solve_demo.jl

# Full MPC simulation
julia --project=. --compile=min --startup-file=no examples/cartpole_example_mpc.jl

# Code generation
julia --project=. --compile=min --startup-file=no examples/cartpole_example_code_generation.jl

# Advanced quadrotor example (sensitivity analysis currently disabled)
julia --project=. --compile=min --startup-file=no examples/quadrotor_hover_codegen.jl
```

**Expected Output:**
- `cartpole_one_solve_demo.jl`: Single optimal control value and predicted trajectory
- `cartpole_example_mpc.jl`: 200-step MPC simulation with convergence in 2-7 iterations
- `cartpole_example_code_generation.jl`: C++ code generated in `examples/out/`

## Running Tests

The test suite verifies all core functionality:

```bash
# Run individual test files
julia --project=. --compile=min --startup-file=no tests/test_basic.jl
julia --project=. --compile=min --startup-file=no tests/test_cache.jl
julia --project=. --compile=min --startup-file=no tests/test_settings.jl

# All basic tests should pass
# Note: Some advanced tests (sensitivity) currently disabled
```

## Advanced Features

### Setup with All Options

```julia
# Setup solver with constraints and advanced settings
u_min = fill(-0.5, 1, N-1)  # Input bounds (nu x N-1)
u_max = fill(0.5, 1, N-1)   

solver = TinyMPCSolver()
status = setup!(solver, A, B, zeros(4), Q, R, rho, 4, 1, N,
               verbose=false,
               abs_pri_tol=1e-4,
               abs_dua_tol=1e-4, 
               max_iter=100,
               adaptive_rho=true,
               adaptive_rho_min=0.1,
               adaptive_rho_max=10.0,
               u_min=u_min,
               u_max=u_max)
```

### Cache Term Management

```julia
# Compute LQR cache matrices
cache = compute_cache_terms(solver, A, B, Q, R, rho=rho)

# Manually set cache terms  
set_cache_terms!(solver, cache.Kinf, cache.Pinf, cache.Quu_inv, cache.AmBKt)
```

### Code Generation

```julia
# Generate standalone C++ code
status = codegen(solver, "output_dir", verbose=true)

# The generated code includes:
# - CMakeLists.txt (build system)
# - setup.py (Python bindings)
# - src/tiny_main.cpp (example usage)
# - src/tiny_data.cpp (problem data)
# - tinympc/tiny_data.hpp (headers)
```

## API Reference

### Core Functions
- `TinyMPCSolver()` - Create solver instance
- `setup!(solver, A, B, f, Q, R, rho, nx, nu, N; kwargs...)` - Initialize solver
- `solve!(solver)` - Solve MPC problem
- `get_solution(solver)` - Get optimal states and controls

### State and Reference Setting
- `set_x0!(solver, x0)` - Set initial state
- `set_x_ref!(solver, x_ref)` - Set state reference trajectory  
- `set_u_ref!(solver, u_ref)` - Set input reference trajectory

### Advanced Features
- `compute_cache_terms(solver, A, B, Q, R; rho)` - Compute LQR cache matrices
- `set_cache_terms!(solver, Kinf, Pinf, Quu_inv, AmBKt)` - Set cache matrices manually
- `update_settings!(solver; kwargs...)` - Update solver settings
- `print_problem_data(solver)` - Print debug information

### Code Generation
- `codegen(solver, output_dir; verbose)` - Generate standalone C++ code

### Setup Options
All parameters supported by Python/MATLAB wrappers:
- `abs_pri_tol`, `abs_dua_tol` - Convergence tolerances (default: 1e-3)
- `max_iter` - Maximum iterations (default: 100)
- `x_min`, `x_max`, `u_min`, `u_max` - State/input bounds 
- `adaptive_rho`, `adaptive_rho_min`, `adaptive_rho_max` - Adaptive rho settings
- `verbose` - Enable verbose output

## Dependencies

### Required
- Julia ≥ 1.6
- LinearAlgebra.jl (standard library)
- ForwardDiff.jl (for automatic differentiation - disabled currently)

### Build Requirements
- CMake ≥ 3.10
- C++ compiler with C++17 support (GCC, Clang, or Apple Clang)
- Git (for submodules)