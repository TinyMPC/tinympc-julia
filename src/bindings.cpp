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

// Get iterations (with null pointer check)
extern "C" int get_iterations() {
    try {
        if (!g_solver || !g_solver->solution) {
            return -1;
        }
        
        return g_solver->solution->iter;
        
    } catch (const std::exception& e) {
        std::cerr << "get_iterations failed: " << e.what() << std::endl;
        return -1;
    }
}

// Check if solved (with null pointer check)
extern "C" int is_solved() {
    try {
        if (!g_solver || !g_solver->solution) {
            return 0;
        }
        
        return g_solver->solution->solved ? 1 : 0;
        
    } catch (const std::exception& e) {
        std::cerr << "is_solved failed: " << e.what() << std::endl;
        return 0;
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

// Set cone constraints
extern "C" int set_cone_constraints(
        int* Acu_data, int Acu_rows,
        int* qcu_data, int qcu_rows,
        double* cu_data, int cu_rows,
        int* Acx_data, int Acx_rows,
        int* qcx_data, int qcx_rows,
        double* cx_data, int cx_rows,
        int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }

        // Map integer vectors (Eigen::VectorXi)
        Eigen::Map<Eigen::VectorXi> Acu(Acu_data, Acu_rows);
        Eigen::Map<Eigen::VectorXi> qcu(qcu_data, qcu_rows);
        Eigen::Map<Eigen::VectorXi> Acx(Acx_data, Acx_rows);
        Eigen::Map<Eigen::VectorXi> qcx(qcx_data, qcx_rows);

        // Map coefficient vectors (tinyVector)
        Eigen::Map<Eigen::VectorXd> cu_vec(cu_data, cu_rows);
        Eigen::Map<Eigen::VectorXd> cx_vec(cx_data, cx_rows);

        tinyVector cu_tiny = cu_vec.cast<tinytype>();
        tinyVector cx_tiny = cx_vec.cast<tinytype>();
        VectorXi Acu_tiny = Acu.cast<int>();
        VectorXi qcu_tiny = qcu.cast<int>();
        VectorXi Acx_tiny = Acx.cast<int>();
        VectorXi qcx_tiny = qcx.cast<int>();

        int status = tiny_set_cone_constraints(g_solver.get(), Acu_tiny, qcu_tiny, cu_tiny, Acx_tiny, qcx_tiny, cx_tiny);

        if (verbose) {
            std::cout << "Set cone constraints status: " << status << std::endl;
        }

        return status;
    } catch (const std::exception& e) {
        std::cerr << "set_cone_constraints failed: " << e.what() << std::endl;
        return -1;
    }
}

// Set linear constraints
extern "C" int set_linear_constraints(
        double* Alin_x_data, int Alin_x_rows, int Alin_x_cols,
        double* blin_x_data, int blin_x_rows,
        double* Alin_u_data, int Alin_u_rows, int Alin_u_cols,
        double* blin_u_data, int blin_u_rows,
        int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }

        // Map matrices/vectors
        Eigen::Map<Eigen::MatrixXd> Alin_x(Alin_x_data, Alin_x_rows, Alin_x_cols);
        Eigen::Map<Eigen::VectorXd> blin_x(blin_x_data, blin_x_rows);
        Eigen::Map<Eigen::MatrixXd> Alin_u(Alin_u_data, Alin_u_rows, Alin_u_cols);
        Eigen::Map<Eigen::VectorXd> blin_u(blin_u_data, blin_u_rows);

        tinyMatrix Alin_x_tiny = Alin_x.cast<tinytype>();
        tinyVector blin_x_tiny = blin_x.cast<tinytype>();
        tinyMatrix Alin_u_tiny = Alin_u.cast<tinytype>();
        tinyVector blin_u_tiny = blin_u.cast<tinytype>();

        int status = tiny_set_linear_constraints(g_solver.get(), Alin_x_tiny, blin_x_tiny, Alin_u_tiny, blin_u_tiny);

        if (verbose) {
            std::cout << "Set linear constraints status: " << status << std::endl;
        }

        return status;
    } catch (const std::exception& e) {
        std::cerr << "set_linear_constraints failed: " << e.what() << std::endl;
        return -1;
    }
}

// Update solver settings (partial)
extern "C" int update_settings(double abs_pri_tol, double abs_dua_tol,
                                int max_iter, int check_termination,
                                int en_state_bound, int en_input_bound,
                                int en_state_soc, int en_input_soc,
                                int en_state_linear, int en_input_linear,
                                int verbose) {
    try {
        if (!g_solver) {
            throw std::runtime_error("Solver not initialized");
        }

        TinySettings* settings = g_solver->settings;
        int status = tiny_update_settings(settings,
                                          (tinytype)abs_pri_tol,
                                          (tinytype)abs_dua_tol,
                                          max_iter,
                                          check_termination,
                                          en_state_bound,
                                          en_input_bound,
                                          en_state_soc,
                                          en_input_soc,
                                          en_state_linear,
                                          en_input_linear);
        if (verbose) {
            std::cout << "Update settings status: " << status << std::endl;
        }
        return status;
    } catch (const std::exception& e) {
        std::cerr << "update_settings failed: " << e.what() << std::endl;
        return -1;
    }
}

// Alias codegen_with_sensitivity to default codegen (sensitivity handled at higher level)
extern "C" int codegen_with_sensitivity(const char* output_dir,
                                         double* dK_data, int dK_rows, int dK_cols,
                                         double* dP_data, int dP_rows, int dP_cols,
                                         double* dC1_data, int dC1_rows, int dC1_cols,
                                         double* dC2_data, int dC2_rows, int dC2_cols,
                                         int verbose) {
    // Currently sensitivity matrices are not used at the C++ layer.
    // Just call the regular codegen for now.
    return codegen(output_dir, verbose);
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