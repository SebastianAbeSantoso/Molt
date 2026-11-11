---
inclusion: always
---

# Repository Structure

Everything lives in one public repo with `.kiro/` at the root. There is no git submodule.

```
Molt/
├── .kiro/
│   ├── specs/molt/          requirements.md, design.md, tasks.md
│   ├── steering/            product, tech, structure, research-integrity, handoff (manual)
│   └── hooks/               post-task checks: checks-after-task.json (sh), checks-after-task-windows.json + scripts/checks.ps1 (PowerShell)
├── docs/
│   ├── elastic-kiroween-handoff.md   source of truth from planning
│   ├── feasibility-report.md         Task 1 output
│   ├── baseline-report.md            Task 4 output; the only source of thresholds
│   ├── kiroween-rules.md             Devpost rules check
│   └── results/                      harness outputs with machine manifests
├── spikes/                  Task 1 throwaway experiments (width emulation, chunked moe, ballast probe)
├── engine/
│   ├── LLAMA_CPP_COMMIT     pinned upstream commit
│   ├── patches/             small patch series: loader, graph, decode loop, server endpoints
│   ├── molt/                new C++: chunked MoE graph, expert pager, control endpoint, tests
│   └── build.py             fetch, patch, build (cpu / cuda / vulkan)
├── tools/ballast/           molt-ballast: VRAM and RAM fillers with a control port
├── convert/                 importance scoring and Elastic Package builder
├── molt/                    Python package (molt-core)
│   ├── models.py            pydantic data models
│   ├── defaults.py          all "initial value" tunables
│   ├── package.py           package validator
│   ├── profiler.py          hardware profiler
│   ├── adapters/            EngineAdapter protocol + molt, stock llama.cpp, edge0, sim
│   ├── controller/          memory model, joint controller, objectives
│   ├── monitor.py           runtime monitor, decision log
│   ├── conversation.py      conversation store (survives engine crashes)
│   └── server.py            FastAPI app
├── harness/
│   ├── evalset/             frozen ids, sampling.json, licences
│   ├── scorers/             humaneval, gsm8k, perplexity
│   └── run.py               config matrix runner, reports
├── web/                     React + Vite + TS chat and dashboard
├── tests/                   Python unit and property tests (mirror molt/ and harness/)
├── pyproject.toml
└── README.md
```

## Conventions

- Python modules are `snake_case`. Tests are `tests/test_<module>.py`. Property tests are `tests/props/test_<topic>.py`.
- C++ in `engine/molt/` uses the `molt_` prefix for public symbols and follows the llama.cpp style.
- Patch files are numbered and named by hook point, for example `0001-loader-molt-chunks.patch`, with one concern per patch.
- Spike code in `spikes/` is not imported by production code. Promote it into `convert/`, `engine/molt/`, or `tools/` when a task says so.
- Results and reports are Markdown plus CSV/JSON in `docs/`. Never put raw model files in the repo.
