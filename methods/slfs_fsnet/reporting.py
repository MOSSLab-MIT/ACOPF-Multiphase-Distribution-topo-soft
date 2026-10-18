"""Console reporting for SLFS/FSNet test metrics."""

from typing import Any, Dict


def print_slfs_test_metrics(batch_size: int, test_metrics: Dict[str, Any]) -> None:
    print(
        f"    Batch size: {test_metrics['batch_size']} | Num batches: {test_metrics['num_batches']} | "
        f"Num samples: {test_metrics['num_samples']}"
    )
    print(
        f"    Inverse time: {test_metrics['total_inverse_time']:.4f}s | "
        f"Avg/batch: {test_metrics['avg_inverse_time_per_batch']:.6f}s"
    )
    print(
        f"    Forward time: {test_metrics['total_forward_time']:.4f}s | "
        f"Avg/batch: {test_metrics['avg_forward_time_per_batch']:.6f}s"
    )
    print(
        f"    Prediction time: {test_metrics['total_prediction_time']:.4f}s | "
        f"Avg/batch: {test_metrics['avg_prediction_time_per_batch']:.6f}s"
    )
    if "gpu_memory" in test_metrics:
        from core.utils.runtime import format_gpu_memory_line

        for device_name, stats in test_metrics["gpu_memory"].items():
            line = format_gpu_memory_line(device_name, stats)
            if line is not None:
                print(f"    {line}")
    print(
        f"    Cost: {test_metrics['cost']:.6f} | Distance: {test_metrics['distance']:.6f} | "
        f"Violation: {test_metrics['violation']:.6e} | Max Vio: {test_metrics['max_violation']:.6e}"
    )
    if "voltage_angle_violation" in test_metrics:
        print(
            "    Components - "
            f"Vmag: {test_metrics['voltage_magnitude_violation']:.6e} "
            f"(max {test_metrics['max_voltage_magnitude_violation']:.6e}) | "
            f"Angle: {test_metrics['voltage_angle_violation']:.6e} "
            f"(max {test_metrics['max_voltage_angle_violation']:.6e}) | "
            f"Branch I: {test_metrics['branch_current_violation']:.6e} "
            f"(max {test_metrics['max_branch_current_violation']:.6e})"
        )
    print(
        f"    Vio Target: {test_metrics['violation_target']:.6e} | Max Vio Target: "
        f"{test_metrics['max_violation_target']:.6e} | |CostErr| {test_metrics['abs_cost_error']:.6f} "
        f"(signed {test_metrics['cost_error']:+.6f}, max {test_metrics['max_abs_cost_error']:.6f}) | "
        f"OptGap: {test_metrics['optimality_gap']:.6%}"
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
