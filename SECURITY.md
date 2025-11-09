# Security & Closed Components

The closed-loop self-improvement stack that powers Hive AI remains private to
preserve alignment controls and prevent uncontrolled replication of
self-modifying intelligence.

## Intentionally Closed Modules

- `brain/learn/` — meta-learning pipelines, policy retraining, and autonomous
  curriculum harvesters
- `brain/core/brain.py` — the primary execution loop that orchestrates
  autonomous plans end-to-end
- Curiosity-driven task generation and exploration heuristics
- Production training traces, preference logs, and uplift manifests

These components enable autonomous self-improvement and are retained in private
infrastructure so we can enforce audits, guardrails, and kill-switches before
promoting new capabilities.

## Responsible Disclosure

If you discover a vulnerability in the open modules, please disclose it
responsibly to security@hiveai.dev. Do not attempt to reconstruct or distribute
closed components; doing so risks bypassing the safety measures that allow us to
ship this project openly.
