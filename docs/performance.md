# Performance Contract

The forecasting engine should keep high-volume numerical work out of Python loops when
possible. The current hot path is simulation draw generation, so binary two-option race
draws are produced through a Numba-accelerated parallel kernel with a Python fallback.

## Configuration

Performance settings live in `configs/model.yaml`:

```yaml
performance:
  engine: numba
  parallel: true
  numba_threads: 0
  benchmark_draws: 10000
  benchmark_repeats: 3
```

- `engine: numba` requests the accelerated kernel.
- `parallel: true` allows parallel Numba execution.
- `numba_threads: 0` keeps Numba's default thread count; set a positive integer to cap
  threads for reproducible benchmarking or constrained machines.
- `benchmark_draws` and `benchmark_repeats` control the benchmark CLI defaults.

## Commands

```bash
uv run civic-signal benchmark run --as-of 2026-05-08 --run-id perf
```

The benchmark writes:

```text
artifacts/benchmarks/<run_id>/performance_benchmark.json
```

The benchmark isolates simulation throughput. It uses deterministic Kalman polling to
construct the setup ensemble, then times repeated simulation draws with the configured
Numba or Python backend. Use forecast or verification commands when measuring the
Bayesian/NUTS sampler itself.

Forecast runs also write:

```text
artifacts/runs/<run_id>/performance.json
```

That file records the requested engine, actual engine, parallel mode, Numba availability,
thread count, and simulation count. `reward_card.json` includes
`R12_performance_contract`. If Numba is requested but unavailable on the platform, the
reward accepts a recorded Python fallback rather than failing spuriously; if Numba is
available, the actual engine must be `numba`.

## Engineering Rules

- Prefer Polars/DuckDB vectorized operations for table transforms.
- Prefer Numba kernels for repeated numerical loops over draws/races.
- Keep fallback Python behavior for unsupported platforms.
- Benchmark before and after major modeling changes that touch simulation, scoring, or
  large feature generation.
- Do not trade off forecast correctness, provenance, or sparse-race honesty for speed.
