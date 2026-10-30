#!/usr/bin/env python3
"""Print a grid YAML config (configs/<grid>.yaml) as bash variable assignments.

    eval "$(python scripts/slurm_runs/grid_config.py <grid> [<method_family>])"

Defines CASE_DIR, DATA, SEED_ARRAY (Slurm array spec of `seed`), SBATCH_OPTS,
TEST_CORE_H5, TEST_FULL_H5 and, when a method family is given, GRID_ARGS
(= args merged with method_args[family]).
"""

import shlex
import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
KEYS = {"case_dir", "seed", "slurm", "args", "method_args", "test_sets"}


def bash_array(name, items):
    return f"{name}=({' '.join(shlex.quote(str(item)) for item in items)})"


def to_cli(args):
    cli = []
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                cli.append(f"--{key}")
        elif isinstance(value, list):
            cli += [f"--{key}", *map(str, value)]
        else:
            cli += [f"--{key}", str(value)]
    return cli


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: grid_config.py <grid> [<method_family>]")
    grid, family = sys.argv[1], (sys.argv[2] if len(sys.argv) == 3 else None)
    path = CONFIG_DIR / f"{grid}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
        sys.exit(f"ERROR: no grid config {path} (available: {' '.join(available)})")
    cfg = yaml.safe_load(path.read_text())
    unknown = set(cfg) - KEYS
    missing = KEYS - set(cfg)
    if unknown or missing:
        sys.exit(f"ERROR: {path}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")

    seeds = cfg["seed"]
    valid = isinstance(seeds, list) and seeds and all(type(s) is int and s >= 0 for s in seeds)
    if not valid or len(set(seeds)) != len(seeds):
        sys.exit(f"ERROR: {path}: seed must be a non-empty list of distinct non-negative ints, got {seeds!r}")

    case_dir = cfg["case_dir"]
    data = f"datagen_topo/datasets/{case_dir}"
    lines = [
        f"CASE_DIR={shlex.quote(case_dir)}",
        f"DATA={shlex.quote(data)}",
        f"SEED_ARRAY={','.join(map(str, seeds))}",
        bash_array("SBATCH_OPTS", [f"--{k.replace('_', '-')}={v}" for k, v in cfg["slurm"].items()]),
    ]
    for name in ("core", "full"):
        tests = [f"{data}/{case_dir}_test_{t}.h5" for t in cfg["test_sets"][name]]
        lines.append(bash_array(f"TEST_{name.upper()}_H5", tests))
    if family is not None:
        if family not in cfg["method_args"]:
            sys.exit(f"ERROR: {path}: method_args has no entry for {family!r}")
        args = {**cfg["args"], **cfg["method_args"][family]}  # method_args override args
        lines.append(bash_array("GRID_ARGS", to_cli(args)))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
