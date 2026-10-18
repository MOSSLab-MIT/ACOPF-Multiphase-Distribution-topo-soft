"""Configuration contract for SLFS and feasibility-seeking."""

from typing import NamedTuple

from core.grids.power_flow import PowerFlowSolver


class FeasibilityConfig(NamedTuple):
    """Hyperparameters for SLFS / PDHG and truncated backprop."""

    fs_tol: float
    fs_maxiter: int
    lin_steps: int
    diff_steps: int
    fs_rho: float
    fs_damp: float = 1.0
    power_flow_solver: str = PowerFlowSolver.FIXED_POINT.value
    voltage_weight: float = 1.0
    fs_step_fraction: float = 0.99
    n_voltage_rows: int = 0
    # If False, bypass SLFS projection entirely: forward pass uses raw NN output as final.
    use_feasibility: bool = True

    def validate(self) -> "FeasibilityConfig":
        """Assert algorithmic invariants and return ``self`` for fluent construction."""
        if self.diff_steps > self.lin_steps:
            raise ValueError(
                f"diff_steps ({self.diff_steps}) must be <= lin_steps ({self.lin_steps})"
            )
        if self.fs_tol <= 0:
            raise ValueError(f"fs_tol must be > 0, got {self.fs_tol}")
        if self.fs_maxiter < 1:
            raise ValueError(f"fs_maxiter must be >= 1, got {self.fs_maxiter}")
        if not (0.0 < self.fs_damp <= 1.0):
            raise ValueError(f"fs_damp must be in (0, 1], got {self.fs_damp}")
        if self.voltage_weight < 1.0:
            raise ValueError(f"voltage_weight must be >= 1.0, got {self.voltage_weight}")
        if self.fs_step_fraction <= 0.0:
            raise ValueError(f"fs_step_fraction must be > 0, got {self.fs_step_fraction}")
        if self.n_voltage_rows < 0:
            raise ValueError(f"n_voltage_rows must be >= 0, got {self.n_voltage_rows}")
        valid_pf = {m.value for m in PowerFlowSolver}
        if self.power_flow_solver not in valid_pf:
            raise ValueError(
                f"power_flow_solver must be one of {sorted(valid_pf)}, "
                f"got {self.power_flow_solver!r}"
            )
        return self

    def overridden(self, **kwargs) -> "FeasibilityConfig":
        """Return a validated copy; keyword values of ``None`` are skipped."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return self._replace(**clean).validate() if clean else self
