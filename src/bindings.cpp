#include <iostream>
#include <memory>
#include <cstring>

// Include Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// Include TinyMPC headers  
#include "tinympc/tiny_api.hpp"
#include "tinympc/types.hpp"
#include "tinympc/codegen.hpp"

// Global solver pointer (like MATLAB bindings)
static std::unique_ptr<TinySolver> g_solver = nullptr;

// Global problem dimensions for array extraction
static int g_nx = 0, g_nu = 0, g_N = 0;

// Setup function - initialize the solver (matches Python PyTinySolver constructor)
extern "C" int setup_solver(double* A_data, int A_rows, int A_cols,
                            double* B_data, int B_rows, int B_cols,
                            double* fdyn_data, int fdyn_rows, int fdyn_cols,
                            double* Q_data, int Q_rows, int Q_cols,
                            double* R_data, int R_rows, int R_cols,
                            double rho, int nx, int nu, int N,
                            double* x_min_data, int x_min_rows, int x_min_cols,
                            double* x_max_data, int x_max_rows, int x_max_cols,
                            double* u_min_data, int u_min_rows, int u_min_cols,
                            double* u_max_data, int u_max_rows, int u_max_cols,
                            int verbose) {
    try {
        if (verbose) {
            std::cout << "Setting up TinyMPC solver with nx=" << nx 
                     << ", nu=" << nu << ", N=" << N << ", rho=" << rho << std::endl;
        }
        
        // Store problem dimensions globally
        g_nx = nx;
        g_nu = nu;
        g_N = N;
        
        // Convert arrays to Eigen matrices
        Eigen::Map<Eigen::MatrixXd> A(A_data, A_rows, A_cols);
        Eigen::Map<Eigen::MatrixXd> B(B_data, B_rows, B_cols);
        Eigen::Map<Eigen::MatrixXd> fdyn(fdyn_data, fdyn_rows, fdyn_cols);
        Eigen::Map<Eigen::MatrixXd> Q(Q_data, Q_rows, Q_cols);
        Eigen::Map<Eigen::MatrixXd> R(R_data, R_rows, R_cols);
        Eigen::Map<Eigen::MatrixXd> x_min(x_min_data, x_min_rows, x_min_cols);
        Eigen::Map<Eigen::MatrixXd> x_max(x_max_data, x_max_rows, x_max_cols);
        Eigen::Map<Eigen::MatrixXd> u_min(u_min_data, u_min_rows, u_min_cols);
        Eigen::Map<Eigen::MatrixXd> u_max(u_max_data, u_max_rows, u_max_cols);
        
        // Convert to tinyMatrix (matching Python PyTinySolver constructor)
        tinyMatrix A_tiny = A.cast<tinytype>();
        tinyMatrix B_tiny = B.cast<tinytype>();
        tinyMatrix fdyn_tiny = fdyn.cast<tinytype>();
        tinyMatrix Q_tiny = Q.cast<tinytype>();
        tinyMatrix R_tiny = R.cast<tinytype>();
        
        // Setup solver (exactly like Python PyTinySolver constructor)
        TinySolver* solver_ptr = nullptr;
        int status = tiny_setup(&solver_ptr, A_tiny, B_tiny, fdyn_tiny, Q_tiny, R_tiny,
                               (tinytype)rho, nx, nu, N, verbose);
        
        if (status != 0) {
            throw std::runtime_error("tiny_setup failed with status " + std::to_string(status));
        }
        
        // Set bounds (exactly like Python PyTinySolver constructor)
        tinyMatrix x_min_tiny = x_min.cast<tinytype>();
        tinyMatrix x_max_tiny = x_max.cast<tinytype>();
        tinyMatrix u_min_tiny = u_min.cast<tinytype>();
        tinyMatrix u_max_tiny = u_max.cast<tinytype>();
        
        if (status == 0) {
            status = tiny_set_bound_constraints(solver_ptr, x_min_tiny, x_max_tiny, u_min_tiny, u_max_tiny);
        }
        
        if (status != 0) {
            if (solver_ptr) {
                delete solver_ptr;
            }
            throw std::runtime_error("Bound constraints setup failed with status " + std::to_string(status));
        }
        
        // Store solver (transfer ownership)
        g_solver.reset(solver_ptr);
        
        if (verbose) {
            std::cout << "TinyMPC solver setup completed successfully" << std::endl;
        }
        
        return 0; // Return status (0 for success)
        
    } catch (const std::exception& e) {
        std::cerr << "Setup failed: " << e.what() << std::endl;
        return -1; // Return error status
    }
}

// Set initial state
extern "C" int set_x0(double* x0_data, int x0_rows, int x0_cols, int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        Eigen::Map<Eigen::MatrixXd> x0(x0_data, x0_rows, x0_cols);
        tinyVector x0_tiny = x0.cast<tinytype>();
        
        int status = tiny_set_x0(g_solver.get(), x0_tiny);
        
        if (status != 0) {
            throw std::runtime_error("tiny_set_x0 failed with status " + std::to_string(status));
        }
        
        if (verbose) {
            std::cout << "Initial state set" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "set_x0 failed: " << e.what() << std::endl;
        return -1;
    }
}

// Set state reference
extern "C" int set_x_ref(double* x_ref_data, int x_ref_rows, int x_ref_cols, int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        Eigen::Map<Eigen::MatrixXd> x_ref(x_ref_data, x_ref_rows, x_ref_cols);
        tinyMatrix x_ref_tiny = x_ref.cast<tinytype>();
        
        int status = tiny_set_x_ref(g_solver.get(), x_ref_tiny);
        
        if (status != 0) {
            throw std::runtime_error("tiny_set_x_ref failed with status " + std::to_string(status));
        }
        
        if (verbose) {
            std::cout << "State reference set" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "set_x_ref failed: " << e.what() << std::endl;
        return -1;
    }
}

// Set input reference
extern "C" int set_u_ref(double* u_ref_data, int u_ref_rows, int u_ref_cols, int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        Eigen::Map<Eigen::MatrixXd> u_ref(u_ref_data, u_ref_rows, u_ref_cols);
        tinyMatrix u_ref_tiny = u_ref.cast<tinytype>();
        
        int status = tiny_set_u_ref(g_solver.get(), u_ref_tiny);
        
        if (status != 0) {
            throw std::runtime_error("tiny_set_u_ref failed with status " + std::to_string(status));
        }
        
        if (verbose) {
            std::cout << "Input reference set" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "set_u_ref failed: " << e.what() << std::endl;
        return -1;
    }
}

// Solve the MPC problem
extern "C" int solve_mpc(int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        int status = tiny_solve(g_solver.get());
        
        if (verbose) {
            std::cout << "Solve completed with status: " << status << std::endl;
        }
        
        return status;
        
    } catch (const std::exception& e) {
        std::cerr << "solve_mpc failed: " << e.what() << std::endl;
        return -1;
    }
}

// Get solution states - copy to pre-allocated buffer
extern "C" int get_states(double* states_buffer, int* rows, int* cols) {
    try {
        if (!g_solver || !g_solver->solution) {
            return -1;
        }
        
        // Get solution states
        const tinyMatrix& states_tiny = g_solver->solution->x;
        
        // Store dimensions
        *rows = states_tiny.rows();
        *cols = states_tiny.cols();
        
        // Convert to double and copy to buffer
        Eigen::MatrixXd states_double = states_tiny.cast<double>();
        std::memcpy(states_buffer, states_double.data(), (*rows) * (*cols) * sizeof(double));
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "get_states failed: " << e.what() << std::endl;
        return -1;
    }
}

// Get solution controls - copy to pre-allocated buffer
extern "C" int get_controls(double* controls_buffer, int* rows, int* cols) {
    try {
        if (!g_solver || !g_solver->solution) {
            return -1;
        }
        
        // Get solution controls
        const tinyMatrix& controls_tiny = g_solver->solution->u;
        
        // Store dimensions
        *rows = controls_tiny.rows();
        *cols = controls_tiny.cols();
        
        // Convert to double and copy to buffer
        Eigen::MatrixXd controls_double = controls_tiny.cast<double>();
        std::memcpy(controls_buffer, controls_double.data(), (*rows) * (*cols) * sizeof(double));
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "get_controls failed: " << e.what() << std::endl;
        return -1;
    }
}

// Cleanup function
extern "C" void cleanup_solver() {
    g_solver.reset();
    g_nx = g_nu = g_N = 0;
}

// Code generation
extern "C" int codegen(const char* output_dir, int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        int status = tiny_codegen(g_solver.get(), output_dir, verbose);
        
        if (verbose) {
            std::cout << "Code generation completed with status: " << status << std::endl;
        }
        
        return status;
        
    } catch (const std::exception& e) {
        std::cerr << "codegen failed: " << e.what() << std::endl;
        return -1;
    }
} 

// Print problem data for debugging (matching Python implementation)
extern "C" int print_problem_data(int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        std::cout << "solution iter:\n" << g_solver->solution->iter << std::endl;
        std::cout << "solution solved:\n" << g_solver->solution->solved << std::endl;
        std::cout << "solution x:\n" << g_solver->solution->x << std::endl;
        std::cout << "solution u:\n" << g_solver->solution->u << std::endl;

        std::cout << "\n\n\ncache rho: " << g_solver->cache->rho << std::endl;
        std::cout << "cache Kinf:\n" << g_solver->cache->Kinf << std::endl;
        std::cout << "cache Pinf:\n" << g_solver->cache->Pinf << std::endl;
        std::cout << "cache Quu_inv:\n" << g_solver->cache->Quu_inv << std::endl;
        std::cout << "cache AmBKt:\n" << g_solver->cache->AmBKt << std::endl;

        std::cout << "\n\n\nabs_pri_tol: " << g_solver->settings->abs_pri_tol << std::endl;
        std::cout << "abs_dua_tol: " << g_solver->settings->abs_dua_tol << std::endl;
        std::cout << "max_iter: " << g_solver->settings->max_iter << std::endl;
        std::cout << "check_termination: " << g_solver->settings->check_termination << std::endl;
        std::cout << "en_state_bound: " << g_solver->settings->en_state_bound << std::endl;
        std::cout << "en_input_bound: " << g_solver->settings->en_input_bound << std::endl;

        std::cout << "\n\n\nnx: " << g_solver->work->nx << std::endl;
        std::cout << "nu: " << g_solver->work->nu << std::endl;
        std::cout << "x:\n" << g_solver->work->x << std::endl;
        std::cout << "u:\n" << g_solver->work->u << std::endl;
        std::cout << "q:\n" << g_solver->work->q << std::endl;
        std::cout << "r:\n" << g_solver->work->r << std::endl;
        std::cout << "p:\n" << g_solver->work->p << std::endl;
        std::cout << "d:\n" << g_solver->work->d << std::endl;
        std::cout << "v:\n" << g_solver->work->v << std::endl;
        std::cout << "vnew:\n" << g_solver->work->vnew << std::endl;
        std::cout << "z:\n" << g_solver->work->z << std::endl;
        std::cout << "znew:\n" << g_solver->work->znew << std::endl;
        std::cout << "g:\n" << g_solver->work->g << std::endl;
        std::cout << "y:\n" << g_solver->work->y << std::endl;
        std::cout << "Q:\n" << g_solver->work->Q << std::endl;
        std::cout << "R:\n" << g_solver->work->R << std::endl;
        std::cout << "Adyn:\n" << g_solver->work->Adyn << std::endl;
        std::cout << "Bdyn:\n" << g_solver->work->Bdyn << std::endl;
        std::cout << "x_min:\n" << g_solver->work->x_min << std::endl;
        std::cout << "x_max:\n" << g_solver->work->x_max << std::endl;
        std::cout << "u_min:\n" << g_solver->work->u_min << std::endl;
        std::cout << "u_max:\n" << g_solver->work->u_max << std::endl;
        std::cout << "Xref:\n" << g_solver->work->Xref << std::endl;
        std::cout << "Uref:\n" << g_solver->work->Uref << std::endl;
        std::cout << "Qu:\n" << g_solver->work->Qu << std::endl;
        std::cout << "primal_residual_state:\n" << g_solver->work->primal_residual_state << std::endl;
        std::cout << "primal_residual_input:\n" << g_solver->work->primal_residual_input << std::endl;
        std::cout << "dual_residual_state:\n" << g_solver->work->dual_residual_state << std::endl;
        std::cout << "dual_residual_input:\n" << g_solver->work->dual_residual_input << std::endl;
        std::cout << "status:\n" << g_solver->work->status << std::endl;
        std::cout << "iter:\n" << g_solver->work->iter << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "print_problem_data failed: " << e.what() << std::endl;
        return -1;
    }
}

// Set cache terms manually (matching Python implementation)
extern "C" int set_cache_terms(double* Kinf_data, int Kinf_rows, int Kinf_cols,
                                double* Pinf_data, int Pinf_rows, int Pinf_cols,
                                double* Quu_inv_data, int Quu_inv_rows, int Quu_inv_cols,
                                double* AmBKt_data, int AmBKt_rows, int AmBKt_cols,
                                int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        if (!g_solver->cache) {
            throw std::runtime_error("Solver cache not initialized");
        }
        
        // Convert arrays to Eigen matrices
        Eigen::Map<Eigen::MatrixXd> Kinf(Kinf_data, Kinf_rows, Kinf_cols);
        Eigen::Map<Eigen::MatrixXd> Pinf(Pinf_data, Pinf_rows, Pinf_cols);
        Eigen::Map<Eigen::MatrixXd> Quu_inv(Quu_inv_data, Quu_inv_rows, Quu_inv_cols);
        Eigen::Map<Eigen::MatrixXd> AmBKt(AmBKt_data, AmBKt_rows, AmBKt_cols);
        
        // Set cache terms
        g_solver->cache->Kinf = Kinf.cast<tinytype>();
        g_solver->cache->Pinf = Pinf.cast<tinytype>();
        g_solver->cache->Quu_inv = Quu_inv.cast<tinytype>();
        g_solver->cache->AmBKt = AmBKt.cast<tinytype>();
        g_solver->cache->C1 = Quu_inv.cast<tinytype>();  // Cache terms
        g_solver->cache->C2 = AmBKt.cast<tinytype>();    // Cache terms
        
        if (verbose) {
            std::cout << "Cache terms set with norms:" << std::endl;
            std::cout << "Kinf norm: " << Kinf.norm() << std::endl;
            std::cout << "Pinf norm: " << Pinf.norm() << std::endl;
            std::cout << "Quu_inv norm: " << Quu_inv.norm() << std::endl;
            std::cout << "AmBKt norm: " << AmBKt.norm() << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "set_cache_terms failed: " << e.what() << std::endl;
        return -1;
    }
}

// Set sensitivity matrices for adaptive rho support (matching Python implementation)
extern "C" int set_sensitivity_matrices(double* dK_data, int dK_rows, int dK_cols,
                                         double* dP_data, int dP_rows, int dP_cols,
                                         double* dC1_data, int dC1_rows, int dC1_cols,
                                         double* dC2_data, int dC2_rows, int dC2_cols,
                                         double rho, int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        // Convert arrays to Eigen matrices
        Eigen::Map<Eigen::MatrixXd> dK(dK_data, dK_rows, dK_cols);
        Eigen::Map<Eigen::MatrixXd> dP(dP_data, dP_rows, dP_cols);
        Eigen::Map<Eigen::MatrixXd> dC1(dC1_data, dC1_rows, dC1_cols);
        Eigen::Map<Eigen::MatrixXd> dC2(dC2_data, dC2_rows, dC2_cols);
        
        // Store sensitivity matrices in the solver's cache
        if (g_solver->cache != nullptr) {
            // For now, we'll just store them for code generation
            if (verbose) {
                std::cout << "Sensitivity matrices set for code generation" << std::endl;
                std::cout << "dK norm: " << dK.norm() << std::endl;
                std::cout << "dP norm: " << dP.norm() << std::endl;
                std::cout << "dC1 norm: " << dC1.norm() << std::endl;
                std::cout << "dC2 norm: " << dC2.norm() << std::endl;
            }
        } else {
            if (verbose) {
                std::cout << "Warning: Cache not initialized, sensitivity matrices will only be used for code generation" << std::endl;
            }
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "set_sensitivity_matrices failed: " << e.what() << std::endl;
        return -1;
    }
}

// Code generation with sensitivity matrices (matching Python implementation)
extern "C" int codegen_with_sensitivity(const char* output_dir,
                                         double* dK_data, int dK_rows, int dK_cols,
                                         double* dP_data, int dP_rows, int dP_cols,
                                         double* dC1_data, int dC1_rows, int dC1_cols,
                                         double* dC2_data, int dC2_rows, int dC2_cols,
                                         int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        // Convert arrays to Eigen matrices and then to tinyMatrix
        Eigen::Map<Eigen::MatrixXd> dK(dK_data, dK_rows, dK_cols);
        Eigen::Map<Eigen::MatrixXd> dP(dP_data, dP_rows, dP_cols);
        Eigen::Map<Eigen::MatrixXd> dC1(dC1_data, dC1_rows, dC1_cols);
        Eigen::Map<Eigen::MatrixXd> dC2(dC2_data, dC2_rows, dC2_cols);
        
        tinyMatrix dK_tiny = dK.cast<tinytype>();
        tinyMatrix dP_tiny = dP.cast<tinytype>();
        tinyMatrix dC1_tiny = dC1.cast<tinytype>();
        tinyMatrix dC2_tiny = dC2.cast<tinytype>();
        
        int status = tiny_codegen_with_sensitivity(g_solver.get(), output_dir,
                                                  &dK_tiny, &dP_tiny, &dC1_tiny, &dC2_tiny, verbose);
        
        if (verbose) {
            std::cout << "Code generation with sensitivity completed with status: " << status << std::endl;
        }
        
        return status;
        
    } catch (const std::exception& e) {
        std::cerr << "codegen_with_sensitivity failed: " << e.what() << std::endl;
        return -1;
    }
}

// Compute cache terms using C++ API
extern "C" int compute_cache_terms(int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        if (!g_solver->cache) {
            throw std::runtime_error("Solver cache not initialized");
        }
        
        int status = tiny_precompute_and_set_cache(g_solver->cache,
                                                   g_solver->work->Adyn,
                                                   g_solver->work->Bdyn,
                                                   g_solver->work->fdyn,
                                                   g_solver->work->Q,
                                                   g_solver->work->R,
                                                   g_solver->work->nx,
                                                   g_solver->work->nu,
                                                   g_solver->cache->rho,
                                                   verbose);
        
        if (verbose) {
            std::cout << "Cache computation completed with status: " << status << std::endl;
        }
        
        return status;
        
    } catch (const std::exception& e) {
        std::cerr << "compute_cache_terms failed: " << e.what() << std::endl;
        return -1;
    }
}

// Enhanced update_settings with all parameters (matching Python implementation)
extern "C" int update_settings(double abs_pri_tol, double abs_dua_tol,
                                int max_iter, int check_termination,
                                int en_state_bound, int en_input_bound,
                                int en_state_soc, int en_input_soc,
                                int en_state_linear, int en_input_linear,
                                int adaptive_rho, double adaptive_rho_min,
                                double adaptive_rho_max, int adaptive_rho_enable_clipping,
                                int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }

        TinySettings* settings = g_solver->settings;
        if (!settings) {
            throw std::runtime_error("Settings not initialized");
        }
        
        // Update all settings
        settings->abs_pri_tol = (tinytype)abs_pri_tol;
        settings->abs_dua_tol = (tinytype)abs_dua_tol;
        settings->max_iter = max_iter;
        settings->check_termination = check_termination;
        settings->en_state_bound = en_state_bound;
        settings->en_input_bound = en_input_bound;
        settings->en_state_soc = en_state_soc;
        settings->en_input_soc = en_input_soc;
        settings->en_state_linear = en_state_linear;
        settings->en_input_linear = en_input_linear;
        
        // Update adaptive rho settings
        settings->adaptive_rho = adaptive_rho;
        settings->adaptive_rho_min = (tinytype)adaptive_rho_min;
        settings->adaptive_rho_max = (tinytype)adaptive_rho_max;
        settings->adaptive_rho_enable_clipping = adaptive_rho_enable_clipping;
        
        if (verbose) {
            std::cout << "Updated settings with adaptive rho support" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "update_settings failed: " << e.what() << std::endl;
        return -1;
    }
}

// Set bound constraints after setup
extern "C" int set_bound_constraints(double* x_min_data, int x_min_rows, int x_min_cols,
                                      double* x_max_data, int x_max_rows, int x_max_cols,
                                      double* u_min_data, int u_min_rows, int u_min_cols,
                                      double* u_max_data, int u_max_rows, int u_max_cols,
                                      int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        Eigen::Map<Eigen::MatrixXd> x_min(x_min_data, x_min_rows, x_min_cols);
        Eigen::Map<Eigen::MatrixXd> x_max(x_max_data, x_max_rows, x_max_cols);
        Eigen::Map<Eigen::MatrixXd> u_min(u_min_data, u_min_rows, u_min_cols);
        Eigen::Map<Eigen::MatrixXd> u_max(u_max_data, u_max_rows, u_max_cols);

        tinyMatrix x_min_tiny = x_min.cast<tinytype>();
        tinyMatrix x_max_tiny = x_max.cast<tinytype>();
        tinyMatrix u_min_tiny = u_min.cast<tinytype>();
        tinyMatrix u_max_tiny = u_max.cast<tinytype>();

        int status = tiny_set_bound_constraints(g_solver.get(), x_min_tiny, x_max_tiny, u_min_tiny, u_max_tiny);
        if (verbose) {
            std::cout << "set_bound_constraints status: " << status << std::endl;
        }
        return status;
    } catch (const std::exception& e) {
        std::cerr << "set_bound_constraints failed: " << e.what() << std::endl;
        return -1;
    }
} 