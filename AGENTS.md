# Repository Guidelines

## Project Structure & Module Organization

- `README.md`: quickstart workflow (DGX-1 defaults) and pointers to deeper docs.
- `INSTRUCTIONS.md`: end-to-end walkthrough (SLURM → Docker → training/DDP/sweeps → staging).
- `docs/`: cluster reference and guides.
  - `docs/serc-dgx1.md`, `docs/serc-dgxh100.md`: SERC policies, storage, queues, access.
  - `docs/dgxh100-adaptation.md`: how to run this repo on DGX-H100.
  - `docs/troubleshooting.md`: common failure modes and fixes.
- `src/`: Python package (training loop, dataset/model, sweep + aggregation).
- `slurm/`: DGX-1 `sbatch` scripts; DGX-H100 variants live in `slurm/dgxh100/`.
- `scripts/`: helper scripts for sanity checks and stage-in/out.
- `configs/`: sweep grid definitions (e.g., `configs/grid.json`).

## Build, Test, and Development Commands

- Build container (recommended path):  
  `docker build -f Dockerfile.modern --build-arg UID=$(id -u) --build-arg GID=$(id -g) --build-arg USERNAME=$USER -t $USER/dgx-demo:torch .`
- Basic Python sanity (no GPU required):  
  `python3 -m py_compile src/*.py`
- Cluster smoke test (DGX-1): `sbatch slurm/00_test_container_1gpu.sbatch`  
  Cluster smoke test (DGX-H100): `sbatch slurm/dgxh100/00_test_container_1gpu.sbatch`

## Coding Style & Naming Conventions

- Python: 4-space indentation; keep code simple and readable (PEP 8-ish).
- Prefer descriptive names; keep functions small and single-purpose.
- Paths/docs: prefer **kebab-case** (`serc-dgx1.md`) over underscores; keep wording professional (no snark).

## Testing Guidelines

- No formal test suite yet. Changes must at least pass `python3 -m py_compile src/*.py`.
- For behavior changes, run the smallest relevant SLURM script and include the command/output location in the PR.

## Commit & Pull Request Guidelines

- This checkout may not include git history; use imperative, scoped commit subjects (e.g., `docs: add DGX-H100 notes`).
- PRs: describe the user-facing change, note which cluster(s) it affects (DGX-1 vs DGX-H100), and update docs + links together.

## Docs & Policy Updates

- When updating SERC reference docs, keep “as published” facts accurate, and bump the “Last updated” date in the document.
