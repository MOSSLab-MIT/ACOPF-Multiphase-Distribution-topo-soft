"""Console reporting for penalty-direct test metrics."""

from typing import Any, Dict

from core.grids.metrics import format_soft_penalty_summary


def print_penalty_direct_test_metrics(batch_size: int, test_metrics: Dict[str, Any]) -> None:
    print(f"    Batch size: {batch_size}")
    print(
        f"    Cost: {test_metrics['cost']:.6f} | Vio: {test_metrics['violation']:.6e} | "
        f"Max Vio: {test_metrics['max_violation']:.6e}"
    )
    print(
        f"    Vio Target: {test_metrics['violation_target']:.6e} | Max Vio Target: "
        f"{test_metrics['max_violation_target']:.6e} | |CostErr| {test_metrics['abs_cost_error']:.6f} "
        f"(signed {test_metrics['cost_error']:+.6f}, max {test_metrics['max_abs_cost_error']:.6f}) | "
        f"OptGap: {test_metrics['optimality_gap']:.6%}"
    )
    if "penalty_v" in test_metrics:
        print(f"    {format_soft_penalty_summary(test_metrics)}")
    print(
        f"    Inverse time: {test_metrics['total_inverse_time']:.4f}s | "
        f"Avg/batch: {test_metrics['avg_inverse_time_per_batch']:.6f}s"
    )
    print(
        f"    Speedup: {test_metrics['speedup']:.2f}x "
        f"(Solve time: {test_metrics['total_solve_time']:.4f}s vs "
        f"Pred time: {test_metrics['total_prediction_time']:.6f}s)"
    )
    print(
        f"    Pder norms - L1: {test_metrics['pder_l1_norm']:.6e} | "
        f"L2: {test_metrics['pder_l2_norm']:.6e} | L∞: {test_metrics['pder_l_inf_norm']:.6e}"
    )
    print(
        f"    Qder norms - L1: {test_metrics['qder_l1_norm']:.6e} | "
        f"L2: {test_metrics['qder_l2_norm']:.6e} | L∞: {test_metrics['qder_l_inf_norm']:.6e}"
    )
