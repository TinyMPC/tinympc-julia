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

// Global solver pointer
static std::unique_ptr<TinySolver> g_solver = nullptr;

// Global problem dimensions for array extraction
static int g_nx = 0, g_nu = 0, g_N = 0;

// Setup function - initialize the solver
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
        
        // Create solver pointer for tiny_setup
        TinySolver* solver_ptr = nullptr;
        
        // Initialize solver with problem data using correct API
        int status = tiny_setup(&solver_ptr,
                                A.cast<tinytype>(),
                                B.cast<tinytype>(),
                                fdyn.cast<tinytype>(),
                                Q.cast<tinytype>(),
                                R.cast<tinytype>(),
                                static_cast<tinytype>(rho),
                                nx, nu, N, verbose);
        
        if (status != 0) {
            return status;
        }
        
        // Set bound constraints using correct API
        status = tiny_set_bound_constraints(solver_ptr,
                                           x_min.cast<tinytype>(),
                                           x_max.cast<tinytype>(),
                                           u_min.cast<tinytype>(),
                                           u_max.cast<tinytype>());
        
        if (status != 0) {
            delete solver_ptr;
            return status;
        }
        
        // Store solver (transfer ownership)
        g_solver.reset(solver_ptr);
        
        if (verbose) std::cout << "TinyMPC solver setup completed successfully" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "setup_solver failed: " << e.what() << std::endl;
        g_solver.reset();
        return -1;
    }
}

extern "C" int set_x0(double* x0_data, int x0_rows, int x0_cols, int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        Eigen::Map<Eigen::MatrixXd> x0(x0_data, x0_rows, x0_cols);
        tinyVector x0_tiny = x0.cast<tinytype>();
        
        int status = tiny_set_x0(g_solver.get(), x0_tiny);
        if (status != 0) {
            throw std::runtime_error("tiny_set_x0 failed");
        }
        
        if (verbose) std::cout << "Initial state set" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "set_x0 failed: " << e.what() << std::endl;
        return -1;
    }
}

extern "C" int set_x_ref(double* x_ref_data, int x_ref_rows, int x_ref_cols, int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        Eigen::Map<Eigen::MatrixXd> x_ref(x_ref_data, x_ref_rows, x_ref_cols);
        tinyMatrix x_ref_tiny = x_ref.cast<tinytype>();
        
        int status = tiny_set_x_ref(g_solver.get(), x_ref_tiny);
        if (status != 0) {
            throw std::runtime_error("tiny_set_x_ref failed");
        }
        
        if (verbose) std::cout << "State reference set" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "set_x_ref failed: " << e.what() << std::endl;
        return -1;
    }
}

extern "C" int set_u_ref(double* u_ref_data, int u_ref_rows, int u_ref_cols, int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        Eigen::Map<Eigen::MatrixXd> u_ref(u_ref_data, u_ref_rows, u_ref_cols);
        tinyMatrix u_ref_tiny = u_ref.cast<tinytype>();
        
        int status = tiny_set_u_ref(g_solver.get(), u_ref_tiny);
        if (status != 0) {
            throw std::runtime_error("tiny_set_u_ref failed");
        }
        
        if (verbose) std::cout << "Input reference set" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "set_u_ref failed: " << e.what() << std::endl;
        return -1;
    }
}

extern "C" int solve_mpc(int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        int status = tiny_solve(g_solver.get());
        
        if (verbose) std::cout << "Solve completed with status: " << status << std::endl;
        return status;
        
    } catch (const std::exception& e) {
        std::cerr << "solve_mpc failed: " << e.what() << std::endl;
        return -1;
    }
}

extern "C" int get_states(double* states_buffer, int* rows, int* cols) {
    try {
        if (!g_solver || !g_solver->solution) {
            return -1;
        }
        
        const tinyMatrix& states_tiny = g_solver->solution->x;
        
        *rows = states_tiny.rows();
        *cols = states_tiny.cols();
        
        Eigen::MatrixXd states_double = states_tiny.cast<double>();
        std::memcpy(states_buffer, states_double.data(), (*rows) * (*cols) * sizeof(double));
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "get_states failed: " << e.what() << std::endl;
        return -1;
    }
}

extern "C" int get_controls(double* controls_buffer, int* rows, int* cols) {
    try {
        if (!g_solver || !g_solver->solution) {
            return -1;
        }
        
        const tinyMatrix& controls_tiny = g_solver->solution->u;
        
        *rows = controls_tiny.rows();
        *cols = controls_tiny.cols();
        
        Eigen::MatrixXd controls_double = controls_tiny.cast<double>();
        std::memcpy(controls_buffer, controls_double.data(), (*rows) * (*cols) * sizeof(double));
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "get_controls failed: " << e.what() << std::endl;
        return -1;
    }
}

extern "C" void cleanup_solver() {
    g_solver.reset();
    g_nx = g_nu = g_N = 0;
}

extern "C" int codegen(const char* output_dir, int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        int status = tiny_codegen(g_solver.get(), output_dir, verbose);
        
        if (verbose) std::cout << "Code generation completed with status: " << status << std::endl;
        return status;
        
    } catch (const std::exception& e) {
        std::cerr << "codegen failed: " << e.what() << std::endl;
        return -1;
    }
} 

// Print problem data for debugging
extern "C" int print_problem_data(int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        // Print key solution and cache information
        std::cout << "=== TinyMPC Problem Data ===" << std::endl;
        std::cout << "Solution: iter=" << g_solver->solution->iter 
                  << ", solved=" << g_solver->solution->solved << std::endl;
        std::cout << "Cache: rho=" << g_solver->cache->rho << std::endl;
        std::cout << "Settings: max_iter=" << g_solver->settings->max_iter 
                  << ", abs_pri_tol=" << g_solver->settings->abs_pri_tol 
                  << ", abs_dua_tol=" << g_solver->settings->abs_dua_tol << std::endl;
        std::cout << "Problem: nx=" << g_solver->work->nx 
                  << ", nu=" << g_solver->work->nu << std::endl;
        
        if (verbose) {
            // Print matrices only if verbose requested
            std::cout << "States x:\n" << g_solver->solution->x << std::endl;
            std::cout << "Controls u:\n" << g_solver->solution->u << std::endl;
            std::cout << "Cache Kinf:\n" << g_solver->cache->Kinf << std::endl;
            std::cout << "Cache Pinf:\n" << g_solver->cache->Pinf << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "print_problem_data failed: " << e.what() << std::endl;
        return -1;
    }
}

// Set cache terms manually
extern "C" int set_cache_terms(double* Kinf_data, int Kinf_rows, int Kinf_cols,
                                double* Pinf_data, int Pinf_rows, int Pinf_cols,
                                double* Quu_inv_data, int Quu_inv_rows, int Quu_inv_cols,
                                double* AmBKt_data, int AmBKt_rows, int AmBKt_cols,
                                int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }
        
        Eigen::Map<Eigen::MatrixXd> Kinf(Kinf_data, Kinf_rows, Kinf_cols);
        Eigen::Map<Eigen::MatrixXd> Pinf(Pinf_data, Pinf_rows, Pinf_cols);
        Eigen::Map<Eigen::MatrixXd> Quu_inv(Quu_inv_data, Quu_inv_rows, Quu_inv_cols);
        Eigen::Map<Eigen::MatrixXd> AmBKt(AmBKt_data, AmBKt_rows, AmBKt_cols);
        
        g_solver->cache->Kinf = Kinf.cast<tinytype>();
        g_solver->cache->Pinf = Pinf.cast<tinytype>();
        g_solver->cache->Quu_inv = Quu_inv.cast<tinytype>();
        g_solver->cache->AmBKt = AmBKt.cast<tinytype>();
        
        if (verbose) {
            std::cout << "Cache terms set - Kinf norm: " << Kinf.norm() 
                      << ", Pinf norm: " << Pinf.norm() << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "set_cache_terms failed: " << e.what() << std::endl;
        return -1;
    }
}



// Code generation with sensitivity matrices
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
        
        Eigen::Map<Eigen::MatrixXd> dK(dK_data, dK_rows, dK_cols);
        Eigen::Map<Eigen::MatrixXd> dP(dP_data, dP_rows, dP_cols);
        Eigen::Map<Eigen::MatrixXd> dC1(dC1_data, dC1_rows, dC1_cols);
        Eigen::Map<Eigen::MatrixXd> dC2(dC2_data, dC2_rows, dC2_cols);
        
        tinyMatrix dK_tiny = dK.cast<tinytype>();
        tinyMatrix dP_tiny = dP.cast<tinytype>();
        tinyMatrix dC1_tiny = dC1.cast<tinytype>();
        tinyMatrix dC2_tiny = dC2.cast<tinytype>();
        
        int status = tiny_codegen_with_sensitivity(g_solver.get(),
                                                   output_dir,
                                                   &dK_tiny, &dP_tiny, &dC1_tiny, &dC2_tiny,
                                                   verbose);
        
        if (verbose) std::cout << "Code generation with sensitivity completed with status: " << status << std::endl;
        return status;
        
    } catch (const std::exception& e) {
        std::cerr << "codegen_with_sensitivity failed: " << e.what() << std::endl;
        return -1;
    }
}



// Enhanced update_settings with all parameters
extern "C" int update_settings(double abs_pri_tol, double abs_dua_tol,
                                int max_iter, int check_termination,
                                int en_state_bound, int en_input_bound,
                                int adaptive_rho, double adaptive_rho_min,
                                double adaptive_rho_max, int adaptive_rho_enable_clipping,
                                int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }

        // Direct update settings on solver (same as MATLAB implementation)
        g_solver->settings->abs_pri_tol = static_cast<tinytype>(abs_pri_tol);
        g_solver->settings->abs_dua_tol = static_cast<tinytype>(abs_dua_tol);
        g_solver->settings->max_iter = max_iter;
        g_solver->settings->check_termination = check_termination;
        g_solver->settings->en_state_bound = en_state_bound;
        g_solver->settings->en_input_bound = en_input_bound;
        
        // Update adaptive rho settings
        g_solver->settings->adaptive_rho = adaptive_rho;
        g_solver->settings->adaptive_rho_min = static_cast<tinytype>(adaptive_rho_min);
        g_solver->settings->adaptive_rho_max = static_cast<tinytype>(adaptive_rho_max);
        g_solver->settings->adaptive_rho_enable_clipping = adaptive_rho_enable_clipping;
        
        int status = 0; // Success
        
        if (verbose) std::cout << "Updated settings with status: " << status << std::endl;
        return status;
        
    } catch (const std::exception& e) {
        std::cerr << "update_settings failed: " << e.what() << std::endl;
        return -1;
    }
}

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
        
        int status = tiny_set_bound_constraints(g_solver.get(),
                                               x_min.cast<tinytype>(),
                                               x_max.cast<tinytype>(),
                                               u_min.cast<tinytype>(),
                                               u_max.cast<tinytype>());
        
        if (verbose) std::cout << "set_bound_constraints status: " << status << std::endl;
        return status;
        
    } catch (const std::exception& e) {
        std::cerr << "set_bound_constraints failed: " << e.what() << std::endl;
        return -1;
    }
} 