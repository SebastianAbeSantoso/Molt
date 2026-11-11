# Implementation Plan — Molt (codename ELASTIC)

## How to read this plan

- Tasks run in order. Task 1 is a gate: no runtime code (Tasks 3 onward) starts until the Feasibility Report is written and the builder approves the route.
- **[HW-L]** means the step runs on the laptop (RTX 4060 8 GB, i9-14980HX, 16 GB RAM, Windows). **[HW-PC]** means the PC (RX 6650 XT 8 GB, i3-12100F, 32 GB RAM). The Kiro sandbox has no GPU. For these steps Kiro prepares the script and exact commands, the builder runs them and pastes the output, and Kiro records the results.
- **Checkpoint** marks a stop for builder review.
- No numeric threshold appears in this plan. Thresholds come from the Baseline Report (Task 4) or from the builder's judgement at a checkpoint.
- Every third-party claim recorded in a task must link its primary source or be marked *unverified*.

## Pre-flight (admin, not coding; do early)

- [ ] P1. Request the Kiroween Kiro credits before December 1, 2026, 12:00 PM PT (December 2, 04:00 WITA). _Requirements: 15.5_
- [ ] P2. Make the GitHub repo `SebastianAbeSantoso/Molt` public, or confirm it will be public before submission. Confirm `.kiro/` is committed and not in `.gitignore`. _Requirements: 15.1_
- [ ] P3. Re-check the official Kiroween 2026 rules on Devpost. Record any difference from the handoff dates in `docs/kiroween-rules.md`. _Requirements: 15.4, 15.6_

## Tasks

- [ ] 1. Model Feasibility Round (gates G0–G4)
  - Output: `docs/feasibility-report.md`, with one section per gate, a candidate matrix, a recommended route, and every unverified claim marked.
  - _Requirements: 1.1–1.9_

  - [ ] 1.1 G0 Novelty: read the full CRISP paper and the related work
    - Read CRISP (arXiv 2610.37442) in full. Answer: does CRISP change width or loading schedule *online* when the memory budget changes during serving? Does it have a user-facing objective? Is it single-user consumer hardware or edge servers? Which models and hardware did it use?
    - Re-read WiSP (arXiv 2606.21868) on MV-WSA: what exactly does its live controller resize, and does it ever change computation?
    - Verify DyMoE (arXiv 2603.19172) and say what it adapts (precision, orchestration) and whether it is online.
    - Verify the handoff's figures for CRISP, D2MoE, and MC-MoE against their papers, and record links.
    - Write the resulting novelty claim. If any work already does the narrowed claim, narrow it again and stop for the builder.
    - _Requirements: 1.6, 1.8, 12.1–12.5_

  - [ ] 1.2 G1 Model: verify Qwen3-30B-A3B
    - Record the Hugging Face repo, licence, and `config.json` values (layers, hidden size, experts, top-K, expert intermediate size, shared experts, attention heads). Compare them with the table in design.md.
    - Find an existing Q8_0 GGUF, or confirm that `convert_hf_to_gguf.py` at the pinned commit supports the `qwen3moe` arch. Record file sizes for Q8_0 and Q4_K_M.
    - Check whether "Qwen3.6-35B-A3B" exists as a public checkpoint. Record the result only; do not plan around it.
    - _Requirements: 1.3, 1.9_

  - [ ] 1.3 Candidate matrix: MoSE, HydraLLM, Edge0, llama.cpp MoE offload
    - For each candidate, record the repo link, licence, checkpoint availability and quality evidence, and nested-width support (existing / needs conversion / not feasible).
    - For each engine, record whether it can run variable-width experts and whether it supports dynamic expert placement across VRAM, RAM, and SSD.
    - MoSE: are trained checkpoints released, at what sizes, and under what licence? This is the G3 fallback only.
    - Edge0: confirm the Windows backends (Vulkan/CPU), the licence, and the supported model list for `edge0-infer`.
    - _Requirements: 1.2–1.4, 1.6_

  - [ ] 1.4 llama.cpp: pin a commit and answer the existing-feature questions
    - Pick a recent llama.cpp commit that builds with CUDA on Windows, and record it in the report. It becomes `engine/LLAMA_CPP_COMMIT` in Task 2.
    - Does the backend scheduler already copy only the *used* experts when host-resident MoE weights are offloaded for a GPU `MUL_MAT_ID`? Cite the code path or PR.
    - Does `llama-imatrix` record per-expert input-activation statistics for `ffn_down_exps`, in a form that gives per-neuron `mean(h²)` per expert?
    - Is `llama_memory_seq_rm` (or its equivalent at this commit) available for removing a partial KV position?
    - _Requirements: 1.4, 1.7_

  - [ ] 1.5 G3 Width quality, emulated without engine changes [HW-L]
    - Write `spikes/width_emulation.py` using `gguf-py`. It loads a Q8_0 GGUF, computes importance per expert neuron from the imatrix and down-projection column norms, and permutes neurons by importance. For each `k ∈ {1, 2}` it zeroes the gate and up rows of neurons beyond `k·256`, which exactly emulates width `k` because `silu(0)·x = 0`. It writes one GGUF per `k`.
    - Control: also write a `k = 1` model that keeps a *random* 256 neurons instead of the top 256. This shows whether importance ordering matters.
    - The builder runs stock `llama-perplexity` and a 5-coding + 5-reasoning smoke subset of the eval set at k = 3, 2, 1, and the random k = 1 control.
    - Record perplexity and solved counts. **Checkpoint:** the builder judges whether `k = 1` and `k = 2` are useful. If not, record the C = 4 fallback plan; do not run it unless asked.
    - _Requirements: 1.5, 4.2, 11.3_

  - [ ] 1.6 G2a Engine spike: chunked `mul_mat_id` on CUDA [HW-L]
    - Write a small standalone C++ test against the pinned ggml in `spikes/chunked_moe/`. It builds an expert FFN as 3 chunks (`[n_embd, 256, n_expert]` gate/up, `[256, n_embd, n_expert]` down) in Q4_K, runs `mul_mat_id` on CUDA for k = 1..3, and compares with an f32 reference.
    - Record correctness and per-layer time per `k` against one fused `mul_mat_id` at full width.
    - _Requirements: 1.4, 1.7_

  - [ ] 1.7 G2b Engine spike: per-MoE-layer host split for slot remapping [HW-L]
    - Extend the spike. After top-K, force a CPU op that reads the expert ids, remaps them to slot ids in a small slot pool, and copies one missing page per layer from pinned host memory. Run it for 48 layers.
    - Measure the per-token overhead of the 48 splits and syncs with no misses, and the cost per missed page, at k = 1..3.
    - Compare with stock `--n-cpu-moe` decode tok/s on the real model at the same budget.
    - **Checkpoint:** the builder decides whether the overhead is acceptable. If not, evaluate Route B (Edge0-infer fork) in 1.9 before deciding.
    - _Requirements: 1.4, 1.7_

  - [ ] 1.8 G4 Budget enforcement on Windows [HW-L]
    - Write `spikes/ballast_probe.cu`: allocate and touch N GB of VRAM, then hold it.
    - With the probe holding 2 and 4 GB, run stock llama.cpp at full offload. Record whether it fails, falls back to shared system memory, or the probe's memory gets demoted. Use Task Manager "Dedicated/Shared GPU memory" and NVML.
    - Record the NVIDIA "CUDA – Sysmem Fallback Policy" setting and its effect. Pick the setting used for all runs.
    - Check the RAM filler approach (`VirtualAlloc` + `VirtualLock` with a raised working-set limit) on the PC [HW-PC], and record whether it reduces available RAM as seen by `psutil`.
    - _Requirements: 2.4, 2.6, 7.8_

  - [ ] 1.9 Edge0 reference run, and Route B check if needed [HW-L]
    - Build `edge0-infer` for Windows and try to run `edge0-35b` on the laptop. Record the backend used, whether it ran, TTFT, tok/s, and the memory it reports.
    - Only if G2 failed: assess how width chunks would fit Edge0's slot design and its Recover-LoRA adapters, and estimate the effort.
    - _Requirements: 1.3, 2.1_

  - [ ] 1.10 Write the Feasibility Report and recommend a route
    - Give pass or fail per gate, with evidence. Name one route and one model. State the chosen engine (A or B) and why. Give the final novelty claim. List open risks.
    - **Checkpoint:** the builder approves the route before Task 2.
    - _Requirements: 1.1, 1.5, 1.7_

- [ ] 2. Repository scaffolding
  - [ ] 2.1 Create the layout from `structure.md`: `molt/`, `engine/molt/`, `engine/patches/`, `engine/LLAMA_CPP_COMMIT`, `tools/ballast/`, `convert/`, `harness/`, `web/`, `tests/`, `docs/`.
  - [ ] 2.2 Set up the Python project with `uv`, Python 3.12, FastAPI, pydantic, pytest, hypothesis, ruff, and `pyproject.toml` with the ruff and pytest config.
  - [ ] 2.3 Add an engine build script that fetches llama.cpp at the pinned commit, applies `engine/patches/*.patch`, adds `engine/molt/` to the build, and builds with CUDA (Windows) or CPU (CI).
  - [ ] 2.4 Set up the web app with Vite, React, TypeScript, vitest, and eslint. FastAPI serves the build output as static files.
  - [ ] 2.5 Add GitHub Actions CI: ruff and pytest; a CPU-only engine build and CTest; web lint and vitest.
  - _Requirements: 15.1_

- [ ] 3. Fixed evaluation set and machine manifest
  - [ ] 3.1 Freeze the eval set: 20 HumanEval ids and 20 GSM8K test ids from a seeded draw, plus 64 windows of 512 tokens from the wikitext-2 test split. Store them in `harness/evalset/` with source links and licences.
  - [ ] 3.2 Write the scorers: sandboxed HumanEval execution with a timeout (pass@1), GSM8K final-number exact match, and the perplexity driver. Report solved counts (n/20), not only percentages.
  - [ ] 3.3 Fix the sampling settings: greedy, fixed `max_tokens`, fixed seed, Qwen3 chat template with thinking off. Store them in `harness/evalset/sampling.json`.
  - [ ] 3.4 Write the machine manifest collector: GPU name and power limit, CPU model and threads, power mode, total and available RAM, SSD model and interface, OS and driver versions, engine version.
  - [ ] 3.5 Unit tests for the scorers on fixed sample outputs.
  - _Requirements: 2.3, 2.7, 10.4, 10.7_

- [ ] 4. Baselines [HW-L, then HW-PC for BL-2 and BL-3 only]
  - [ ] 4.1 Write `StockLlamaCppAdapter` and `Edge0Adapter`. Both start a server process, stream generation, and expose perplexity where the engine supports it.
  - [ ] 4.2 Write the harness runner: config matrix (4/6/8 GB VRAM budget via the VRAM probe from 1.8, 2K/4K/8K context, RAM Budget only on the PC), per-run metrics, process I/O counters for SSD bytes, peak VRAM/RAM, minimum available RAM, OOM detection. Output CSV and JSON.
  - [ ] 4.3 Run BL-1 (full offload where it fits), BL-2 (`--n-cpu-moe` with the smallest N that fits), and BL-3 (Edge0, reference point) on the laptop. Label BL-2 with the CPU.
  - [ ] 4.4 Write `docs/baseline-report.md`: performance envelope, OOM cases, the statement that 4/6 GB budgets simulate memory only, and machine manifests.
  - [ ] 4.5 **Checkpoint:** the builder sets the thresholds the later tasks need: the usefulness bar for Req 11.3 and `D_bal` for the Balanced objective. Record them in the Baseline Report.
  - _Requirements: 2.1–2.7, 10.8_

- [ ] 5. Elastic Package conversion pipeline
  - [ ] 5.1 `convert/importance.py`: per-expert neuron importance from imatrix statistics and down-column norms. Promote the logic from spike 1.5.
  - [ ] 5.2 `convert/build_package.py`: permute, split into C = 3 chunks (`blk.{L}.ffn_{gate,up,down}_exps.c{j}`), copy non-expert tensors, write `molt.*` metadata, and quantize with `llama-quantize` using the imatrix.
  - [ ] 5.3 `molt/package.py`: package validator that checks every metadata rule in design.md and names the bad key or tensor on failure.
  - [ ] 5.4 Validator tests on synthetic GGUF headers (missing chunk, wrong chunk size, top-K mismatch).
  - [ ] 5.5 Build the Qwen3-30B-A3B package [HW-L or a machine with enough disk]. Record size, calibration data hash, and conversion time.
  - _Requirements: 4.1–4.4_

- [ ] 6. molt-engine: width control (enables configs A and B)
  - [ ] 6.1 Patch the loader to recognise `molt.*` keys and chunk tensors for `qwen3moe`.
  - [ ] 6.2 Patch the graph builder to replace the fused MoE FFN with the chunk loop over `j < k`, applying router weights once after the chunk sum.
  - [ ] 6.3 Add the control endpoint (`/molt/config`, `/molt/state`, `/molt/telemetry`) and the `PendingConfig` atomic swap, applied at Safe Boundaries (between decode steps, between prefill micro-batches, prefill to decode). Emit `config_applied`.
  - [ ] 6.4 Add per-token telemetry: latency, VRAM used, k.
  - [ ] 6.5 Engine test: chunk-loop output equals the f32 prefix-width reference for k = 1..3, and k = 3 matches the original fused expert within quant tolerance (Property 10).
  - [ ] 6.6 Engine test: `k` changes only inside boundary-apply (Property 4).
  - [ ] 6.7 [HW-L] Check that full-width perplexity of the package matches the original model within noise, and run the perplexity smoke test at k = 1, 2.
  - _Requirements: 4.4, 5.1, 5.2, 5.4, 5.5, 7.3_

- [ ] 7. molt-engine: Expert Pager (enables configs C and D)
  - [ ] 7.1 Per-layer VRAM slot pools with resize (shrink first). Page = `(L, e, j)`.
  - [ ] 7.2 Slot lookup: a host op after top-K remaps expert ids to slot ids and issues demand loads for missing pages `j < k` only.
  - [ ] 7.3 Working-set tracker: per-layer ring buffer over `W` tokens, `|WS_L|` at several windows, `ws_overflow` signal.
  - [ ] 7.4 Eviction: non-working-set pages first (LRU), working-set pages only after `ws_overflow`. Count thrash events.
  - [ ] 7.5 RAM cache (pinned host buffers) capped by RAM Budget, with working-set-then-LRU over `W_ram`.
  - [ ] 7.6 SSD reader: unbuffered aligned reads from the package file (`FILE_FLAG_NO_BUFFERING` on Windows).
  - [ ] 7.7 Per-tier counters: hits, misses, loads, evictions, bytes, miss stall time.
  - [ ] 7.8 Prefetch toggle, off by default: next-layer router guess, lower priority than demand loads.
  - [ ] 7.9 OOM inside a step: abort the step, remove the partial KV position, emit `oom`, wait for a new config.
  - [ ] 7.10 Engine tests on CPU: eviction order (Property 3), working set kept while it fits (Property 2), at most `k` pages per miss (Property 9).
  - _Requirements: 6.1–6.9, 7.9_

- [ ] 8. molt-core: models, profiler, adapters, controller
  - [ ] 8.1 Data models from design.md (`EngineConfig`, `PendingConfig`, `TokenStats`, `Decision`, `Objective`). No token ids, text, or logits in any controller input type.
  - [ ] 8.2 Hardware profiler: engine `--probe`, NVML, `psutil` (total and available RAM), disk free, CPU info, conservative fallbacks marked "assumed".
  - [ ] 8.3 `SimEngineAdapter`: replays a recorded routing trace and models page sizes, slots, PCIe/SSD bandwidth, and per-chunk compute.
  - [ ] 8.4 `MoltLlamaCppAdapter`: process start and stop, generate, post_config, telemetry, perplexity, health.
  - [ ] 8.5 Memory model: `need(k, slots)` from engine-reported allocations, `effective_budget`.
  - [ ] 8.6 Joint Controller, session start: startup probe at k = 1..3, Objective rules (Quality / Balanced with `D_bal` / Speed capped at C − 1), refusal when nothing fits.
  - [ ] 8.7 Joint Controller, mid-session: the narrowing sequence (shrink cold slots → narrow k → trim to minimum → stop and report), widening with `h` and `T_hold`, memory-pressure signals only.
  - [ ] 8.8 Runtime Monitor (250 ms frames) and JSON-lines decision log.
  - [ ] 8.9 Property tests (hypothesis) on `SimEngineAdapter` for Properties 1, 5, 6, 7, and 8.
  - _Requirements: 3.1–3.5, 5.1–5.7, 7.1–7.8, 7.10_

- [ ] 9. Fillers: `molt-ballast`
  - [ ] 9.1 `--vram` (CUDA): allocate in 256 MB blocks, touch, resize on command over a local control port. Promote from spike 1.8.
  - [ ] 9.2 `--ram`: `VirtualAlloc` + `VirtualLock`, resize on command.
  - [ ] 9.3 Harness and API integration: set filler size per run and for the squeeze schedule. Mark runs invalid if the filler cannot reach its target.
  - _Requirements: 2.2, 2.6, 7.4_

- [ ] 10. OOM and crash recovery
  - [ ] 10.1 Conversation store in `molt-core`, independent of the engine process.
  - [ ] 10.2 Restart path: on engine exit, restart at the next narrower config, re-run prefill on history, resume streaming. If the narrowest fails, show the cause and keep history.
  - [ ] 10.3 Test on `SimEngineAdapter` with injected OOM and crash: history unchanged (Property 11).
  - _Requirements: 7.9, 8.6_

- [ ] 11. API and web UI
  - [ ] 11.1 FastAPI: `/api/chat` (SSE), `/api/session` (Objective, budgets, prefetch, thinking toggle default off), `/api/squeeze`, `/api/profile`, `/ws/telemetry`. Bind to 127.0.0.1 only.
  - [ ] 11.2 Chat panel: streaming, Objective selector, VRAM budget slider, thinking toggle.
  - [ ] 11.3 Dashboard: VRAM gauge with budget marker and external usage, 48 × 128 expert heatmap (tier, chunks resident, working-set outline), skin indicator, tok/s and TTFT tiles, decision feed in plain words, working-set curve, "stale" indicator.
  - [ ] 11.4 Offline check: no external assets, bundled fonts and scripts.
  - [ ] 11.5 Component tests (vitest) rendering from recorded telemetry.
  - _Requirements: 8.1–8.6, 9.1–9.6_

- [ ] 12. Evaluation: 2x2, baselines, squeeze [HW-L]
  - [ ] 12.1 Add configs A–D to the harness runner (A: k = 3, residency off; B: adaptive width, residency off; C: k = 3, residency on; D: both).
  - [ ] 12.2 Run the full matrix at 4/6/8 GB × 2K/4K/8K with the fixed eval set. Re-run the baselines at the same commit.
  - [ ] 12.3 Run prefetch on versus off for configs C and D.
  - [ ] 12.4 Run the squeeze scenario (8 → 6 → 4 → 8 GB) on a fixed conversation. Record survival, stall at each change, and tok/s and quality per phase.
  - [ ] 12.5 Report analytic active parameters and FLOPs per token per `k`, checked against measured expert time (Req 11.3).
  - [ ] 12.6 Write `docs/results/` with the performance envelope, solved counts (n/20), and rows where D does not beat B or C shown alongside the rest.
  - [ ] 12.7 **Checkpoint:** the builder reviews against Req 11 success criteria. If D does not beat B or C, the write-up says so and does not claim the advantage.
  - _Requirements: 10.1–10.8, 11.1–11.5_

- [ ] 13. Write-up
  - [ ] 13.1 README: the problem, the demo task, how to run it, and the narrowed novelty claim exactly as approved in Task 1.
  - [ ] 13.2 Prior art section citing WiSP, CRISP, Edge0, Krasis, MoSE, HydraLLM, D2MoE, MC-MoE, DyMoE, Yao et al., and DATS, with links. Every figure verified.
  - [ ] 13.3 "Old ideas in the code" section: link Liu et al. (1991) to the mandatory/optional scheduler and Denning (1968) to the working-set tracker, with code links.
  - [ ] 13.4 Limitations: memory-only tier simulation, small eval set variance, mixed-width KV after a width change, laptop GPU variant, CPU dependence of BL-2.
  - _Requirements: 12.1–12.7_

- [ ] 14. Stretch: AMD on the PC [HW-PC]
  - [ ] 14.1 Build the engine with the Vulkan backend and run the same package. Add a Vulkan path to `molt-ballast --vram`.
  - [ ] 14.2 Run the 2x2 at one budget with the RAM filler on, and record which requirements pass and fail. Do not block the submission on this.
  - _Requirements: 13.1–13.3_

- [ ] 15. Stretch: beyond v1
  - [ ] 15.1 Precision skins: assess nested quantization within the chunk layout. Design note only, unless time remains.
  - [ ] 15.2 DeepSeek V4.1 Flash: only after Req 11.1–11.4 pass. Re-verify all V4.1 figures, assess disk/RAM needs against the test machines, and run as a closing beat if feasible.
  - _Requirements: 4.5, 14.1–14.3_

- [ ] 16. Demo and submission
  - [ ] 16.1 Script the demo: Impossible Model (BL-1 fails or crawls, then Molt runs), Live Squeeze (8 → 6 → 4 GB mid-conversation), quick quality check. Open with the coding task, not a graph.
  - [ ] 16.2 Record the video, under 3 minutes.
  - [ ] 16.3 Final compliance check: public repo, `.kiro/` committed, video link, submission before December 4, 2026, 2:00 PM PT.
  - _Requirements: 15.1–15.4_
