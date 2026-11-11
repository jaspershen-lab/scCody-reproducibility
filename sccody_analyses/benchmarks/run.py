"""Run one of the retained study comparator implementations."""
import importlib
import sys
from .engine.dynamics.source_plan import prepare_source_plan


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"scnode", "scdiffeq", "artemis"}:
        raise SystemExit("Usage: python -m sccody_analyses.benchmarks.run {scnode,scdiffeq,artemis} [arguments]")
    method = sys.argv[1]
    implementation = importlib.import_module(f"sccody_analyses.benchmarks.engine.dynamics.methods.{method}")
    args = implementation.parse_args(sys.argv[2:])
    plan = prepare_source_plan(args, implementation.IDENTITY, validate_input_layer=method != "scnode")
    if not plan.skipped:
        implementation.run_benchmark(args, plan)


if __name__ == "__main__":
    main()
