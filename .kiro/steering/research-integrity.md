---
inclusion: always
---

# Research Integrity Rules

Molt is judged partly on honesty about prior art. These rules apply to code comments, docs, the README, the dashboard text, and the demo script.

## Numbers

- Never invent benchmark numbers, paper results, model capabilities, or thresholds.
- Every number is one of three kinds:
  - **measured:** from a harness run in `docs/results/` or `docs/baseline-report.md`, with a machine manifest;
  - **cited:** from a primary source, with a link, re-verified before it is quoted;
  - **estimate / initial value:** labelled as such, as in design.md.
- Thresholds come only from the Baseline Report or from the builder at a checkpoint.
- The eval set is small (20 coding, 20 reasoning). Report solved counts (n/20) and treat a one- or two-problem difference as noise.
- 4 GB and 6 GB budgets simulate memory limits only, not the compute of weaker GPUs. Say so wherever tiers are compared.
- `--n-cpu-moe` results depend on the CPU. Label them with the CPU and do not compare them across machines as GPU results.

## Claims

- State the novelty claim exactly as in `product.md`, or as narrowed by `docs/feasibility-report.md`.
- Always cite WiSP for working-set MoE paging and CRISP for importance-ordered widths with joint scheduling. Never imply Molt introduced either.
- If configuration D does not beat B or C, say so plainly and do not claim the joint advantage.
- Mark anything not verified from a primary source as *unverified*, and say how it will be checked.

## Sources

- Prefer papers, official repos, and official model cards over blogs.
- Link sources inline. Paraphrase; do not paste long passages.
