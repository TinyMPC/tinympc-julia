#include <jlcxx/jlcxx.hpp>
#include <Eigen/Dense>
#include "tinympc/tiny_api.hpp"
#include "tinympc/types.hpp"
#include "tinympc/codegen.hpp"
#include <memory>
#include <string>
#include <vector>

class JuliaTinySolver {
public:
    JuliaTinySolver() : solver_ptr(nullptr) {}

    int setup(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& fdyn,
              const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, double rho,
              int nx, int nu, int N, 
              const Eigen::MatrixXd& x_min, const Eigen::MatrixXd& x_max,
              const Eigen::MatrixXd& u_min, const Eigen::MatrixXd& u_max, int verbose) {
        tinyMatrix A_tiny = A.cast<tinytype>();
        tinyMatrix B_tiny = B.cast<tinytype>();
        tinyMatrix fdyn_tiny = fdyn.cast<tinytype>();
        tinyMatrix Q_tiny = Q.cast<tinytype>();
        tinyMatrix R_tiny = R.cast<tinytype>();
        tinyMatrix x_min_tiny = x_min.cast<tinytype>();
        tinyMatrix x_max_tiny = x_max.cast<tinytype>();
        tinyMatrix u_min_tiny = u_min.cast<tinytype>();
        tinyMatrix u_max_tiny = u_max.cast<tinytype>();

        int status = tiny_setup(&solver_ptr, A_tiny, B_tiny, fdyn_tiny, Q_tiny, R_tiny, 
                                (tinytype)rho, nx, nu, N, verbose);
        if (status == 0) {
            status = tiny_set_bound_constraints(solver_ptr, x_min_tiny, x_max_tiny, u_min_tiny, u_max_tiny);
        }
        return status;
    }

    int set_x0(const Eigen::MatrixXd& x0) {
        tinyVector x0_tiny = x0.cast<tinytype>();
        return tiny_set_x0(solver_ptr, x0_tiny);
    }

    int set_x_ref(const Eigen::MatrixXd& x_ref) {
        tinyMatrix x_ref_tiny = x_ref.cast<tinytype>();
        return tiny_set_x_ref(solver_ptr, x_ref_tiny);
    }

    int set_u_ref(const Eigen::MatrixXd& u_ref) {
        tinyMatrix u_ref_tiny = u_ref.cast<tinytype>();
        return tiny_set_u_ref(solver_ptr, u_ref_tiny);
    }

    int set_bound_constraints(const Eigen::MatrixXd& x_min, const Eigen::MatrixXd& x_max,
                              const Eigen::MatrixXd& u_min, const Eigen::MatrixXd& u_max) {
        tinyMatrix x_min_tiny = x_min.cast<tinytype>();
        tinyMatrix x_max_tiny = x_max.cast<tinytype>();
        tinyMatrix u_min_tiny = u_min.cast<tinytype>();
        tinyMatrix u_max_tiny = u_max.cast<tinytype>();
        return tiny_set_bound_constraints(solver_ptr, x_min_tiny, x_max_tiny, u_min_tiny, u_max_tiny);
    }

    int solve() {
        return tiny_solve(solver_ptr);
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> get_solution() {
        tinyMatrix x_sol = solver_ptr->solution->x;
        tinyMatrix u_sol = solver_ptr->solution->u;
        return {x_sol.cast<double>(), u_sol.cast<double>()};
    }

    std::vector<double> get_stats() {
        return {
            static_cast<double>(solver_ptr->work->iter),
            static_cast<double>(solver_ptr->work->status),
            solver_ptr->work->primal_residual_state,
            solver_ptr->work->primal_residual_input
        };
    }

    int codegen(const std::string& output_dir, int verbose) {
        return tiny_codegen(solver_ptr, output_dir.c_str(), verbose);
    }

    int codegen_with_sensitivity(const std::string& output_dir,
                                 const Eigen::MatrixXd& dK,
                                 const Eigen::MatrixXd& dP,
                                 const Eigen::MatrixXd& dC1,
                                 const Eigen::MatrixXd& dC2,
                                 int verbose) {
        tinyMatrix dK_copy = dK.cast<tinytype>();
        tinyMatrix dP_copy = dP.cast<tinytype>();
        tinyMatrix dC1_copy = dC1.cast<tinytype>();
        tinyMatrix dC2_copy = dC2.cast<tinytype>();
        return tiny_codegen_with_sensitivity(solver_ptr, output_dir.c_str(),
                                             &dK_copy, &dP_copy, &dC1_copy, &dC2_copy, verbose);
    }

    int set_sensitivity_matrices(const Eigen::MatrixXd& dK,
                                 const Eigen::MatrixXd& dP,
                                 const Eigen::MatrixXd& dC1,
                                 const Eigen::MatrixXd& dC2) {
        if (solver_ptr->cache != nullptr) {
            solver_ptr->cache->dK = dK.cast<tinytype>();
            solver_ptr->cache->dP = dP.cast<tinytype>();
            solver_ptr->cache->dC1 = dC1.cast<tinytype>();
            solver_ptr->cache->dC2 = dC2.cast<tinytype>();
            return 0;
        }
        return -1;
    }

    int set_cache_terms(const Eigen::MatrixXd& Kinf,
                        const Eigen::MatrixXd& Pinf,
                        const Eigen::MatrixXd& Quu_inv,
                        const Eigen::MatrixXd& AmBKt) {
        if (!solver_ptr->cache) return -1;
        solver_ptr->cache->Kinf = Kinf.cast<tinytype>();
        solver_ptr->cache->Pinf = Pinf.cast<tinytype>();
        solver_ptr->cache->Quu_inv = Quu_inv.cast<tinytype>();
        solver_ptr->cache->AmBKt = AmBKt.cast<tinytype>();
        solver_ptr->cache->C1 = Quu_inv.cast<tinytype>();
        solver_ptr->cache->C2 = AmBKt.cast<tinytype>();
        return 0;
    }

    int update_settings(double abs_pri_tol, double abs_dua_tol, int max_iter, int check_termination,
                        int en_state_bound, int en_input_bound, int adaptive_rho,
                        double adaptive_rho_min, double adaptive_rho_max, int adaptive_rho_enable_clipping) {
        if (solver_ptr && solver_ptr->settings) {
            solver_ptr->settings->abs_pri_tol = abs_pri_tol;
            solver_ptr->settings->abs_dua_tol = abs_dua_tol;
            solver_ptr->settings->max_iter = max_iter;
            solver_ptr->settings->check_termination = check_termination;
            solver_ptr->settings->en_state_bound = en_state_bound;
            solver_ptr->settings->en_input_bound = en_input_bound;
            solver_ptr->settings->adaptive_rho = adaptive_rho;
            solver_ptr->settings->adaptive_rho_min = adaptive_rho_min;
            solver_ptr->settings->adaptive_rho_max = adaptive_rho_max;
            solver_ptr->settings->adaptive_rho_enable_clipping = adaptive_rho_enable_clipping;
            return 0;
        }
        return -1;
    }

    void reset() {
        if (solver_ptr) {
            delete solver_ptr;
            solver_ptr = nullptr;
        }
    }

    std::string print_problem_data() {
        std::ostringstream oss;
        if (!solver_ptr) return "Solver not initialized";
        oss << "solution iter: " << solver_ptr->solution->iter << "\n";
        oss << "solution solved: " << solver_ptr->solution->solved << "\n";
        oss << "abs_pri_tol: " << solver_ptr->settings->abs_pri_tol << "\n";
        oss << "abs_dua_tol: " << solver_ptr->settings->abs_dua_tol << "\n";
        oss << "max_iter: " << solver_ptr->settings->max_iter << "\n";
        oss << "check_termination: " << solver_ptr->settings->check_termination << "\n";
        oss << "en_state_bound: " << solver_ptr->settings->en_state_bound << "\n";
        oss << "en_input_bound: " << solver_ptr->settings->en_input_bound << "\n";
        oss << "nx: " << solver_ptr->work->nx << "\n";
        oss << "nu: " << solver_ptr->work->nu << "\n";
        oss << "iter: " << solver_ptr->work->iter << "\n";
        oss << "status: " << solver_ptr->work->status << "\n";
        return oss.str();
    }

    ~JuliaTinySolver() {
        reset();
    }

private:
    TinySolver* solver_ptr;
};

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.add_type<JuliaTinySolver>("JuliaTinySolver")
        .constructor()
        .method("setup", &JuliaTinySolver::setup)
        .method("set_x0", &JuliaTinySolver::set_x0)
        .method("set_x_ref", &JuliaTinySolver::set_x_ref)
        .method("set_u_ref", &JuliaTinySolver::set_u_ref)
        .method("set_bound_constraints", &JuliaTinySolver::set_bound_constraints)
        .method("solve", &JuliaTinySolver::solve)
        .method("get_solution", &JuliaTinySolver::get_solution)
        .method("get_stats", &JuliaTinySolver::get_stats)
        .method("codegen", &JuliaTinySolver::codegen)
        .method("codegen_with_sensitivity", &JuliaTinySolver::codegen_with_sensitivity)
        .method("set_sensitivity_matrices", &JuliaTinySolver::set_sensitivity_matrices)
        .method("set_cache_terms", &JuliaTinySolver::set_cache_terms)
        .method("update_settings", &JuliaTinySolver::update_settings)
        .method("reset", &JuliaTinySolver::reset)
        .method("print_problem_data", &JuliaTinySolver::print_problem_data);
}