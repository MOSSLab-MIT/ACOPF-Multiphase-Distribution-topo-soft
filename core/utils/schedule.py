"""Learning-rate schedule construction and its startup summary."""

import optax


def lr_schedule_params(
    base_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 0.5,
) -> dict[str, float | int]:
    """Resolved warmup-cosine schedule parameters for Optax and startup logging."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    decay_steps = max(total_steps - warmup_steps, 1)
    min_lr = min_lr_ratio * base_lr
    init_lr = 0.0 if warmup_steps > 0 else base_lr
    return {
        "peak_lr": base_lr,
        "init_lr": init_lr,
        "min_lr": min_lr,
        "min_lr_ratio": min_lr_ratio,
        "warmup_epochs": warmup_epochs,
        "warmup_steps": warmup_steps,
        "total_epochs": total_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "decay_steps": decay_steps,
    }


def create_lr_schedule(
    base_lr,
    warmup_epochs,
    total_epochs,
    steps_per_epoch,
    min_lr_ratio: float = 0.5,
):
    """Create warmup-cosine learning-rate schedule."""
    params = lr_schedule_params(
        base_lr, warmup_epochs, total_epochs, steps_per_epoch, min_lr_ratio
    )

    return optax.warmup_cosine_decay_schedule(
        init_value=params["init_lr"],
        peak_value=params["peak_lr"],
        warmup_steps=params["warmup_steps"],
        decay_steps=params["decay_steps"],
        end_value=params["min_lr"],
    )


def print_training_schedule_summary(
    *,
    learning_rate: float,
    warmup_epochs: int,
    epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 0.5,
) -> None:
    """Print resolved LR warmup/cosine schedule (requires ``steps_per_epoch``)."""
    params = lr_schedule_params(
        learning_rate, warmup_epochs, epochs, steps_per_epoch, min_lr_ratio
    )
    print("Training schedule:")
    print(f"  learning_rate (peak): {params['peak_lr']:g}")
    print(
        f"  warmup_epochs: {params['warmup_epochs']} "
        f"({params['warmup_steps']} optimizer steps)"
    )
    print(
        f"  steps_per_epoch: {params['steps_per_epoch']}  |  "
        f"total_epochs: {params['total_epochs']}  |  "
        f"total_steps: {params['total_steps']}"
    )
    print(
        f"  initial_lr: {params['init_lr']:g}  ->  peak_lr: {params['peak_lr']:g}  ->  "
        f"min_lr: {params['min_lr']:g}  (min_lr_ratio={params['min_lr_ratio']})"
    )
    print(f"  cosine_decay_steps: {params['decay_steps']}")
    print()
