"""Console reporting for penalty+FS test metrics."""

from typing import Any, Dict


def print_penalty_fs_test_metrics(batch_size: int, test_metrics: Dict[str, Any]) -> None:
    shared = test_metrics["shared"]
    pred = test_metrics["pred"]
    fs = test_metrics["fs"]
    print(f"    Batch size: {batch_size}")
    print(
        f"    [pred] Cost: {pred['cost']:.6f} | Vio: {pred['violation']:.6e} | Max Vio: "
        f"{pred['max_violation']:.6e} | |CostErr|: {pred['abs_cost_error']:.6f} "
        f"(signed {pred['cost_error']:+.6f}, max {pred['max_abs_cost_error']:.6f}) | "
        f"OptGap: {pred['optimality_gap']:.6%}"
    )
    print(
        f"    [fs]   Cost: {fs['cost']:.6f} | Vio: {fs['violation']:.6e} | Max Vio: "
        f"{fs['max_violation']:.6e} | |CostErr|: {fs['abs_cost_error']:.6f} "
        f"(signed {fs['cost_error']:+.6f}, max {fs['max_abs_cost_error']:.6f}) | "
        f"OptGap: {fs['optimality_gap']:.6%}"
    )
    print(
        f"    Target Vio: {shared['violation_target']:.6e} | Max Target Vio: "
        f"{shared['max_violation_target']:.6e} | Distance(pred,fs): {shared['distance']:.6e}"
    )
    print(
        f"    Raw speedup (NN only): {pred['raw_speedup']:.2f}x | "
        f"FS pipeline speedup: {fs['speedup']:.2f}x"
    )
    print(
        f"    Inverse time: {shared['total_inverse_time']:.4f}s | "
        f"Avg/batch: {shared['avg_inverse_time_per_batch']:.6f}s"
    )
    print(
        f"    Solve time: {shared['total_solve_time']:.4f}s | Raw pred: "
        f"{pred['total_raw_prediction_time']:.6f}s | FS overhead: {fs['total_fs_time']:.6f}s | "
        f"Full pipeline: {fs['total_prediction_time']:.6f}s"
    )
    print(
        f"    [pred] Pder L1/L2/L∞: {pred['pder_l1_norm']:.6e} / "
        f"{pred['pder_l2_norm']:.6e} / {pred['pder_l_inf_norm']:.6e}"
    )
    print(
        f"    [fs]   Pder L1/L2/L∞: {fs['pder_l1_norm']:.6e} / "
        f"{fs['pder_l2_norm']:.6e} / {fs['pder_l_inf_norm']:.6e}"
    )
