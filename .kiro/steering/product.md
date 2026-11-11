---
inclusion: always
---

# Product: Molt (codename ELASTIC)

Kiroween 2026 entry, Resurrection category. The spec is in `.kiro/specs/molt/`. The source of truth is `docs/elastic-kiroween-handoff.md`.

## What it is

Molt is a local inference runtime for one Mixture-of-Experts model (Qwen3-30B-A3B, pending the Feasibility Round). It keeps a conversation running on an 8 GB consumer GPU when the model does not fit in VRAM. It does this by changing two things during a live session:

- **Width ("skin shedding").** Each expert is stored as 3 importance-ordered chunks. Running fewer chunks runs a narrower expert. Chunk 0 is mandatory work, and chunks 1–2 are optional work, following Imprecise Computation (Liu et al., 1991).
- **Residency.** Expert chunks are pages across VRAM, RAM, and SSD. A working-set policy keeps recently used experts in VRAM, following the Working Set Model (Denning, 1968).

## Users

People running local LLMs on Windows PCs with roughly 6–12 GB of VRAM: local AI enthusiasts, developers who want private coding help, and students without a high-end GPU.

## The claim, and only this claim

Molt's contribution is a controller that changes expert width and expert residency together **during a live session when the VRAM budget changes**. A user-chosen objective (Speed, Balanced, Quality) steers it, and it schedules width as mandatory and optional work. Output quality is a controlled variable.

Not claimed:
- Working-set paging for MoE. That is WiSP (Nokia, arXiv 2606.21868), which also has a live residency controller.
- Importance-ordered expert widths, or jointly optimising width and loading. That is CRISP (arXiv 2610.37442).
- SSD expert streaming. That is Edge0.

If the Feasibility Report narrows the claim further, the report wins over this file.

## v1 scope

In scope: hardware profiler, one elastic package with 3 widths, a capacity controller (per-session width, fixed top-K), a working-set residency manager, a joint controller that adapts only at Safe Boundaries under memory pressure, chat, a live dashboard, and an evaluation harness (2x2 plus baselines).

Out of scope: training from scratch, writing an engine from scratch, dynamic top-K, choosing width per token by content, precision skins (stretch), expert dropping, multi-GPU, cloud, CPU-ARC, automatic conversion of arbitrary models.

Stretch: AMD RX 6650 XT via Vulkan; DeepSeek V4.1 Flash as a closing beat only.

## Demo

The demo is under 3 minutes. It opens with a coding task, not a graph.

1. **Impossible Model:** the stock setup fails or crawls on the laptop, then Molt runs the same model and prompt.
2. **Live Squeeze:** the budget drops 8 → 6 → 4 GB mid-conversation, and Molt narrows, evicts, and keeps talking.
3. **Quality check:** the output is still useful.

The dashboard (VRAM gauge, expert heatmap, skin indicator, tok/s) is the star of the demo.

## Deadline

Submission is due December 4, 2026, 2:00 PM PT. The repo must be public with `.kiro/` committed.
