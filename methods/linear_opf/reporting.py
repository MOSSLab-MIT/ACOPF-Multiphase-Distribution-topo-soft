"""Console reporting for linear OPF benchmark metrics."""

from typing import Any

from core.grids.metrics import format_soft_penalty_summary


def print_linear_opf_benchmark_metrics(batch_size: int, metrics: dict[str, Any]) -> None:
    print(
        f"    Batch size: {batch_size} | Num batches: {metrics['num_batches']} | "
        f"Num samples: {metrics['num_samples']} | Evaluated: {metrics.get('num_evaluated', metrics['num_samples'])} | "
        f"QP solved: {metrics['num_qp_solved']} | QP failed: {metrics['num_qp_failed']} | "
        f"Skipped (lin PF): {metrics.get('num_skipped', 0)}"
    )
    print("    Timing breakdown (per sample averages):")
    print(
        f"      Topo inverse: {metrics['avg_inverse_time_per_sample'] * 1000:.2f} ms "
        f"(excluded from speedup)"
    )
    print(
        f"      Linearization: {metrics['avg_linearization_time_per_sample'] * 1000:.2f} ms "
        f"(PF + build A)"
    )
    print(
        f"      Clarabel solver: {metrics['avg_qp_solver_time_per_sample'] * 1000:.2f} ms "
        f"(CVXPY solver_stats; vs Ipopt)"
    )
    print(
        f"      CVXPY overhead: {metrics['avg_qp_overhead_time_per_sample'] * 1000:.2f} ms "
        f"(qp_wall - solver)"
    )
    print(
        f"      QP wall total: {metrics['avg_qp_time_per_sample'] * 1000:.2f} ms "
        f"(lin + sync + solve)"
    )
    print(
        f"      Post-QP PF: {metrics['avg_pf_time_per_sample'] * 1000:.2f} ms "
        f"(metrics only)"
    )
    print(
        f"      Ipopt (HDF5): {metrics['avg_ipopt_solve_time_per_sample'] * 1000:.2f} ms"
    )
    print(
        f"    Cost: {metrics['cost']:.6f} | Distance: {metrics['distance']:.6f} | "
        f"Violation: {metrics['violation']:.6e} | Max Vio: {metrics['max_violation']:.6e}"
    )
    print(
        f"    Vio Target: {metrics['violation_target']:.6e} | Max Vio Target: "
        f"{metrics['max_violation_target']:.6e} | |CostErr| {metrics['abs_cost_error']:.6f} "
        f"(signed {metrics['cost_error']:+.6f}, max {metrics['max_abs_cost_error']:.6f}) | "
        f"OptGap: {metrics['optimality_gap']:.6%}"
    )
    if "penalty_v" in metrics:
        print(f"    {format_soft_penalty_summary(metrics)}")
    print(
        f"    Solver speedup: {metrics['solver_speedup']:.2f}x "
        f"(Ipopt {metrics['total_solve_time']:.4f}s vs "
        f"Clarabel {metrics['total_qp_solver_time']:.4f}s)"
    )
    print(
        f"    Solver+PF speedup: {metrics['solver_pf_speedup']:.2f}x "
        f"(Ipopt {metrics['total_solve_time']:.4f}s vs "
        f"Clarabel+PF {metrics['total_solver_pf_time']:.4f}s)"
    )
    print(
        f"    E2E speedup: {metrics['speedup']:.2f}x "
        f"(Ipopt {metrics['total_solve_time']:.4f}s vs "
        f"QP+PF {metrics['total_e2e_time']:.4f}s; excludes topo inverse)"
    )
