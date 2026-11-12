"""Console reporting for direct-prediction test metrics."""

from typing import Any, Dict

from core.grids.metrics import SOFT_PENALTY_KEYS, format_soft_penalty_summary


def print_direct_prediction_test_metrics(batch_size: int, test_metrics: Dict[str, Any]) -> None:
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
    if "cost_pf" in test_metrics:
        print(
            f"    PF-repaired (PF at predicted DERs): Cost: {test_metrics['cost_pf']:.6f} | "
            f"Vio: {test_metrics['violation_pf']:.6e} | Max Vio: {test_metrics['max_violation_pf']:.6e} | "
            f"|CostErr| {test_metrics['abs_cost_error_pf']:.6f} | "
            f"OptGap: {test_metrics['optimality_gap_pf']:.6%} | Speedup: {test_metrics['speedup_pf']:.2f}x "
            f"(PF time: {test_metrics['total_pf_time']:.6f}s)"
        )
        pf_pen = " / ".join(f"{test_metrics[k + '_pf']:.4e}" for k in SOFT_PENALTY_KEYS)
        print(f"    PF-repaired soft penalty ρ·Σs (V / I / θ / Sub): {pf_pen}")
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
