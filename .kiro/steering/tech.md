---
inclusion: always
---

# Tech Stack and Commands

## Stack

| Part | Technology |
| --- | --- |
| Inference engine | llama.cpp fork at a pinned commit (`engine/LLAMA_CPP_COMMIT`), C++17, CMake. New code in `engine/molt/`, hooks in `engine/patches/*.patch`. Fallback engine: Edge0-infer fork, only if the Feasibility Report chooses it. |
| GPU backends | CUDA (primary, RTX 4060 laptop). Vulkan (stretch, RX 6650 XT PC). CPU backend for CI. |
| Runtime core | Python 3.12, `uv`, FastAPI, pydantic v2, `psutil`, `pynvml` |
| Tests | `pytest`, `hypothesis` (Python); CTest (engine); `vitest` (web) |
| Lint and format | `ruff check`, `ruff format` (Python); `eslint` (web); `clang-format` with the llama.cpp style for `engine/molt/` |
| Web UI | React, Vite, TypeScript, served as static files by FastAPI. No external CDNs: the app must work offline. |
| Model tooling | `gguf-py`, `llama-imatrix`, `llama-quantize`, `llama-perplexity` from the pinned commit |
| CI | GitHub Actions: Python lint and tests, CPU-only engine build and CTest, web lint and tests |

## Common commands

These commands work once Task 2 scaffolding exists.

```sh
uv sync                                  # install Python deps
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                         # unit + property tests (no GPU needed)
uv run python -m molt.server             # start molt-core on 127.0.0.1
cd web && npm ci && npm run lint && npx vitest --run
python engine/build.py --backend cpu     # fetch llama.cpp at pinned commit, patch, build
python engine/build.py --backend cuda    # on the Windows laptop
ctest --test-dir engine/build -L molt
```

Never run watch modes (`vitest` without `--run`, `npm run dev`) in the Kiro sandbox.

## Environment facts

- The Kiro sandbox is Linux with **no GPU**. Anything that needs a GPU is a hardware step: Kiro writes the script and exact commands, the builder runs them on the laptop or PC and pastes the output back.
- The target OS is Windows. Code that touches memory, files, or GPUs must work on Windows first: `FILE_FLAG_NO_BUFFERING`, `VirtualLock`, WDDM behaviour, path handling.
- Test machines:
  - **Laptop:** RTX 4060 8 GB, i9-14980HX, 16 GB RAM, NVMe. Primary target.
  - **PC:** RX 6650 XT 8 GB, i3-12100F, 32 GB RAM, NVMe. Stretch target; uses the RAM filler.
- Use `SimEngineAdapter` to develop and test controller, API, and UI logic without a GPU.

## Engineering rules

- Per-token work lives in C++. Policy lives in Python and reaches the engine only as a `PendingConfig`, applied at the next Safe Boundary. Python never blocks the decode loop.
- The controller never receives token ids, text, or logits. Do not add them to any controller input type.
- Expert top-K is fixed. Width changes only at Safe Boundaries, and only in response to memory pressure.
- Keep patches to llama.cpp minimal. Put logic in `engine/molt/` and keep the patch to the hook points.
- Every tunable in design.md marked **initial value** lives in one config module (`molt/defaults.py`), with a comment pointing to design.md. Do not scatter magic numbers.
- Secrets: none expected. Never commit model weights, tokens, or machine-specific paths.
