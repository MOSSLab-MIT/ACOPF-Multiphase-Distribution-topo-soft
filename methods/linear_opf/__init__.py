"""Test-only linear OPF baseline (fixed-reference CVXPY solve).

Modules:
  test.py      — CLI entrypoint
  runner.py    — benchmark loop over test HDF5 batches
  solver.py    — topology-keyed CVXPY problem cache
  reporting.py — console metric summaries
"""
