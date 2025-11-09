# OpenAGI

openAGI is a curated, production-ready slice of the Hive Brain platform.
It packages the safety-critical observability stack, proof-bundle tooling,
strategy selector, and world-model scaffolding needed to run compliance and
verification workflows without the rest of the private research codebase.

> **Scope guarantee**: no training checkpoints or private strategy assets are
> included. All bundled modules are self-contained and rely only on the
> dependencies listed in `requirements.txt` / `requirements-dev.txt`.

---

## Included components

- **brain.obs** – structured metrics, health checks, and Prometheus-friendly
  exporters (`metrics.py`, `health_checks.py`, `enhanced_observability.py`).
- **brain.meta.strategy_selector** – deterministic task routing logic shared by
  the production Brain.
- **brain.world_model** – light bootstrap registry, simple policy runner, and
  validation harness used by the proof tooling.
- **tools/ci/proof_bundle.py** – full proof bundle generator with all gates and
  artifact collectors (plugins, contradictions, curriculum, retrain execution,
  RBAC, budgets, etc.).
- **brain.consistency.goals / ledger** – contradiction remediation helpers used
  by proof gates and dashboards.
- **brain.io.atomic** – atomic file writers used by most collectors.
- **tests/** – regression coverage for health checks, metrics registry, proof
  bundle minimal runs, goal lifecycle, contradiction collectors, and self-goal
  accounting.

Optional extras (loaded dynamically if available):

- Prometheus client (`prometheus_client`), Redis (`redis`), PostgreSQL
  (`psycopg2`), and psutil. The code degrades gracefully when they are absent.

---

## Quick start

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run the regression suite:

```bash
pytest
```

Generate a proof bundle (writes to `./artifacts` by default):

```bash
python -m tools.ci.proof_bundle
```

The script prints the path to the generated `proof.json` and replicates the
canonical artifact layout used in production deployments.

---

## Artifacts and directories

- `artifacts/` – default root for proof assets, selector logs, curriculum
  snapshots, and retrain scorecards.
- `tests/` – pytest-based regression coverage. Minimal suites avoid heavy
  dependencies by monkeypatching expensive collectors.
- `brain/obs/metrics.py` – in-process metrics registry used both by
  observability and proof gates.

All filesystem writes use atomic patterns (`brain.io.atomic`) so they are safe
for concurrent runners.

---

## Configuration & env vars

Key environment variables consumed by the included modules:

| Variable | Purpose | Default |
| --- | --- | --- |
| `ARTIFACTS_DIR` | Root for proof bundle outputs and collectors | `./artifacts` |
| `PROOF_FAIL_ON_REQUIRED_GATES` | Exit non-zero when required gates fail (auto-disabled under pytest) | `0` |
| `PROOF_CONTRADICTION_MAX` | Maximum allowed contradictions before failing gate | `0` |
| `PROOF_CONSISTENCY_GOALS_MIN` | Minimum required remediation goals when contradictions exist | `0` |
| `BRAIN_EMERGENT_ROUTING` | Enables additional emergence regression suite | `0` |
| `PROOF_SIGN_KEY` | Optional HMAC key for signing proof bundles | _unset_ |

See `tools/ci/proof_bundle.py` for the full list of optional knobs.

---

## Testing philosophy

The bundled tests aim to keep the OSS package deterministic while exercising the
most critical gates:

- Health checks gracefully degrade when optional dependencies are missing.
- Metrics registry snapshots still produce consistent bucket counts.
- Proof bundle smoke tests stub heavy runners but validate gates and signatures.
- Goal lifecycle, self-goal, and contradiction collectors assert artifact
  integrity and gate semantics.

All tests pass with the dependencies specified above; no external services are
required.

---

## Contributing

1. Fork the future `HiveAI-Labs/openAGI` repository.
2. Add regression tests for every change (follow the patterns in `tests/`).
3. Run `ruff` / `mypy` if you add them; keep all code type-hinted.
4. Confirm `pytest` still passes before submitting PRs.

For security-sensitive additions (new collectors, proof gates, or networked
health checks), document the risk considerations in the PR description and add a
regression test showing failure detection.

---

## License

Distributed under the MIT License (see `LICENSE`). Include attribution when
redistributing or embedding these modules downstream.
