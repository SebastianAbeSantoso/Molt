# Requirements Document — Molt (codename ELASTIC)

## Introduction

Molt is a hardware-adaptive local inference runtime for one Mixture-of-Experts (MoE) model package. It runs on consumer Windows PCs with roughly 6–12 GB of GPU memory and keeps a conversation going when the model does not fit in VRAM.

Molt coordinates two mechanisms in one joint controller:

- **Capacity adaptation ("skin shedding").** Each expert executes at one of several nested widths. The scheduling policy follows Imprecise Computation (Liu et al., 1991): a mandatory part always runs, and an optional part runs only when the budget allows.
- **Residency adaptation.** Expert weights are placed across VRAM, system RAM, and SSD. The placement policy follows the Working Set Model (Denning, 1968): experts used in the recent token window stay in VRAM, and the runtime avoids thrashing.

The source of truth for this spec is `docs/elastic-kiroween-handoff.md`.

### Novelty boundary

These requirements apply throughout the spec and the write-up.

- Working-set paging for MoE is prior art. WiSP (Nokia; arXiv 2606.21868, v1 June 2026, v2 August 2026) treats low-resource MoE serving as a working-set problem, keeps outputs byte-identical, and includes MV-WSA, a live controller that resizes the expert cache and KV cache while serving. Molt cites WiSP and does not claim working-set MoE residency, or live residency control, as its contribution.
- Joint optimisation of expert width and expert loading is prior art. CRISP (arXiv 2610.37442, October 2026) reorders neurons by importance so one stored expert supports multiple widths, and co-designs that pruning with runtime scheduling that overlaps expert loading with compute, on memory-constrained edge servers. Molt cites CRISP and does not claim importance-ordered expert widths, or the general idea of optimising width and loading together, as its contribution. Whether CRISP adapts online to a changing memory budget is unverified and is checked in the Feasibility Round.
- Imprecise computation applied to neural networks is prior art in parts. Yao et al. apply it to DNN layers and early exits, and DATS (RTSS 2026) applies it to LLM tokens. Molt does not claim it as its contribution.
- Molt's claimed contribution is limited to: a controller that changes expert width and expert residency together **during a live session when the VRAM budget changes**, steered by a user-selected Objective, with width scheduled as mandatory and optional work in the sense of Liu et al. (1991), for a local, single-user MoE on consumer hardware. Output quality is a controlled variable traded against latency and memory. IF the Feasibility Round finds that CRISP or other work already does this, THEN the claim SHALL be narrowed again before any write-up.

### Hard constraints from the handoff

- Molt builds on an existing inference engine. It does not train a model from scratch or write an inference engine from scratch.
- Expert top-K stays fixed in v1. Width is the capacity control.
- v1 is width-only. Precision skins are a stretch goal. Expert skin (dropping experts per token) is out of scope because it conflicts with fixed top-K.
- Width is selected per session in v1. Per-request selection is an optional later extension.
- Molt never chooses width per token based on token content. The only mid-generation width change allowed is a narrowing or widening at a Safe Boundary (between tokens) in response to memory pressure.
- The primary target is the RTX 4060 8 GB (CUDA). The RX 6650 XT 8 GB (Vulkan) is a stretch compatibility target.
- No numeric performance or quality threshold is fixed before baseline runs exist. Requirements below that need a threshold refer to values recorded in the baseline report.
- CPU-ARC, dynamic top-K, per-token width changes, multi-GPU, cloud inference, and automatic conversion of arbitrary MoE models are out of scope.

## Glossary

- **Molt / Runtime**: the system specified here. "ELASTIC" is the codename used in the handoff.
- **Engine**: the existing third-party inference engine Molt builds on. It is chosen in the Model Feasibility Round.
- **Elastic Model Package (Package)**: a single model artifact on disk whose experts support at least three nested widths.
- **Width / Width Skin**: the fraction of an expert's importance-ordered neurons that execute. Narrower widths are prefixes of wider ones.
- **Precision Skin**: a lower-precision execution of an expert drawn from nested quantization of the same stored weights.
- **Skin Profile**: a named combination of width and, if supported, precision, for example Full, Medium, Slim, Emergency.
- **Mandatory Work**: shared layers, attention, and the narrowest expert slice. It always runs.
- **Optional Work**: wider expert slices, extra precision, and extra context. It runs only when the budget allows.
- **Objective**: the user-selected mode — Speed, Balanced, or Quality — that bounds how much optional work is allowed.
- **VRAM Budget**: the GPU memory limit Molt must stay within. It is either detected or imposed, for example 4, 6, or 8 GB.
- **RAM Budget**: an imposed limit on the system RAM available for warm expert weights, used to force the SSD tier into use on machines with large RAM.
- **Working Set**: the set of distinct experts referenced within the last window of W tokens.
- **Thrashing**: repeated eviction and reload of experts that belong to the working set.
- **Safe Boundary**: the point between two generated tokens (or between prefill and decode) where the Runtime can change width or residency without corrupting generation state. Width changes at a Safe Boundary are allowed only in response to memory pressure. Design defines the exact mechanics.
- **Joint Controller**: the component that chooses the skin profile and residency plan together.
- **Baseline Report**: the recorded measurements from standard setups that set thresholds for later requirements.

## Requirements

### Requirement 1: Model Feasibility Gate

**User Story:** As the builder, I want to confirm where a usable elastic MoE comes from before building the runtime, so that I do not build on a model or engine that cannot support nested widths and dynamic residency.

#### Acceptance Criteria

1. THE project SHALL produce a Feasibility Report before any runtime component is implemented.
2. THE Feasibility Report SHALL evaluate at least MoSE, HydraLLM, Edge0, and llama.cpp MoE expert offload, and SHALL record which Engine (default: llama.cpp fork; fallback: Edge0-infer fork) is chosen and why.
3. FOR EACH candidate, THE Feasibility Report SHALL record the repository link, licence, checkpoint availability, checkpoint quality evidence, and whether nested expert widths are supported by existing code, need conversion, or are not feasible.
4. FOR EACH candidate Engine, THE Feasibility Report SHALL state whether it can execute variable-width experts and whether it supports dynamic expert placement across VRAM, RAM, and SSD.
5. THE Feasibility Report SHALL recommend exactly one route (reproduce, adapt, or convert) and one small MoE model to test skin shedding on first.
6. THE Feasibility Report SHALL mark every claim it could not verify from a primary source as unverified and SHALL state how it will be verified.
7. IF no candidate supports both variable-width experts and dynamic residency, THEN THE Feasibility Report SHALL state the smallest engine modification required and its estimated effort, and the builder SHALL decide whether to proceed before design is finalised.
8. THE Feasibility Report SHALL include a reading of the full CRISP paper stating whether CRISP adapts width or scheduling online in response to a changing memory budget, and SHALL state the resulting novelty claim.
9. THE Feasibility Report SHALL verify that the Qwen3-30B-A3B checkpoint is available and record its licence. The project SHALL NOT plan around any other checkpoint (for example Qwen3.6-35B-A3B) until the report verifies it exists.

### Requirement 2: Baseline Measurement

**User Story:** As the builder, I want baseline numbers from a standard setup, so that later thresholds come from measurements and not guesses.

#### Acceptance Criteria

1. THE project SHALL run the chosen model on the RTX 4060 in these named baselines: (a) llama.cpp with full GPU offload where it fits, (b) llama.cpp with `--n-cpu-moe` static expert offload, and (c) Edge0-infer with its released model. Edge0 runs its own checkpoint, so its results SHALL be reported as a reference point and not as a same-model comparison.
2. THE baseline SHALL cover VRAM budgets of 4, 6, and 8 GB, context lengths of 2K, 4K, and 8K tokens, and chat, coding, and reasoning prompts.
3. FOR EACH baseline run, THE project SHALL record TTFT, generation tok/s, prompt-processing tok/s, peak VRAM, peak system RAM, out-of-memory occurrence, and a quality score on the fixed evaluation set.
4. THE Baseline Report SHALL state that 4 and 6 GB budgets simulate memory limits only and do not simulate the weaker compute of lower-tier GPUs.
5. THE Baseline Report SHALL be the only source for numeric thresholds referenced by later requirements.
6. WHERE a test machine has enough system RAM to hold the whole Package, THE baseline and harness SHALL also run under an imposed RAM Budget so that the SSD tier is exercised, and SHALL report results with and without the RAM Budget.
7. THE Baseline Report SHALL record each test machine's GPU model and power limit, CPU model and thread count, power mode, total system RAM and RAM available after OS overhead, SSD model and interface, OS version, GPU driver version, and Engine version. Results that depend on CPU execution (for example `--n-cpu-moe`) SHALL be labelled with the CPU they ran on.

### Requirement 3: Hardware Profiling

**User Story:** As a local user, I want Molt to detect my hardware on launch, so that I do not have to configure offloading or quantization by hand.

#### Acceptance Criteria

1. WHEN Molt launches, THE Runtime SHALL detect GPU vendor and model, total and free VRAM, total and free system RAM, free SSD space at the Package location, and the available Engine backends.
2. WHEN profiling completes, THE Runtime SHALL display the detected values to the user.
3. WHERE the user sets a VRAM Budget lower than detected free VRAM, THE Runtime SHALL use the user-set budget. WHERE the user sets a RAM Budget, THE Runtime SHALL keep warm expert weights within it.
4. IF a required measurement cannot be obtained, THEN THE Runtime SHALL report which value is missing and SHALL fall back to a conservative default defined in design.
5. WHILE a session is active, THE Runtime SHALL sample free VRAM at an interval defined in design so that external GPU memory use is detected.

### Requirement 4: Elastic Model Package

**User Story:** As a local user, I want one model download that works across hardware configurations, so that I do not need a different file per machine.

#### Acceptance Criteria

1. THE Runtime SHALL load one Package that supports at least three nested expert widths.
2. THE Package SHALL store each expert so that each narrower width is a prefix of the next wider width, with neurons ordered by importance offline.
3. WHEN a Package is imported, THE Runtime SHALL validate its metadata (architecture, expert count, top-K, available widths, available precisions) and SHALL reject a Package with missing or inconsistent metadata with an error that names the problem.
4. THE Runtime SHALL execute all supported widths from the same Package without reloading or converting the Package.
5. (Stretch) WHERE the Package supports nested precision levels, THE Runtime SHALL expose them as Precision Skins. Precision skins are not part of v1.

### Requirement 5: Capacity Controller (Imprecise Computation Scheduler)

**User Story:** As a local user, I want Molt to choose how much of each expert to run, so that I get a usable answer even when resources are tight.

#### Acceptance Criteria

1. THE Runtime SHALL classify each forward pass's work into Mandatory Work and Optional Work.
2. THE Runtime SHALL always execute Mandatory Work.
3. WHEN the user selects an Objective, THE Capacity Controller SHALL bound the allowed Optional Work: Speed allows the least, Quality allows the most, and Balanced lies between them.
4. THE Capacity Controller SHALL select one Skin Profile per session. THE Capacity Controller SHALL NOT choose width per token based on token content, and SHALL change width mid-session only at a Safe Boundary in response to memory pressure (Requirement 7).
5. THE Capacity Controller SHALL keep expert top-K equal to the Package's configured top-K.
6. IF the selected Skin Profile cannot fit within the VRAM Budget together with the Working Set and KV cache, THEN THE Capacity Controller SHALL select the next narrower Skin Profile.
7. IF the narrowest Skin Profile cannot fit, THEN THE Runtime SHALL tell the user that the model cannot run within the budget and SHALL NOT start generation.

### Requirement 6: Residency Manager (Working Set Policy)

**User Story:** As a local user, I want the full expert bank to be usable without it all sitting in VRAM, so that a larger model runs on my GPU.

#### Acceptance Criteria

1. THE Residency Manager SHALL place expert weights in three tiers: VRAM, system RAM, and SSD.
2. THE Residency Manager SHALL track the Working Set as the distinct experts referenced in the last W tokens, where W is defined in design.
3. WHILE the Working Set fits within the VRAM available for experts, THE Residency Manager SHALL NOT evict a member of the Working Set.
4. WHEN an expert outside VRAM is selected by the router, THE Residency Manager SHALL load the slice needed for the current width and SHALL NOT load wider slices than the current width requires.
5. THE Residency Manager SHALL provide expert prefetching as a toggle that is off by default. WHEN prefetching is on, THE harness SHALL measure it against prefetching off. THE design SHALL NOT depend on prefetching for its claimed benefit; the expected lever is that a narrower width moves fewer bytes per expert miss. (WiSP reports that prefetching does not help single-stream decode over PCIe; Edge0 reports gains from routing prediction when streaming from SSD. Both are to be checked by measurement.)
6. WHEN eviction is required, THE Residency Manager SHALL evict experts outside the Working Set before any member of the Working Set.
7. IF the Working Set does not fit within the VRAM available for experts, THEN THE Residency Manager SHALL signal the Joint Controller instead of repeatedly evicting Working Set members.
8. THE Residency Manager SHALL record per-tier hits, misses, loads, evictions, and bytes transferred from SSD and RAM.
9. THE Residency Manager SHALL be switchable off, so that residency can be measured independently of capacity adaptation.

### Requirement 7: Joint Controller and Runtime Monitor

**User Story:** As a local user, I want Molt to react when my GPU memory changes during a conversation, so that the conversation continues instead of crashing.

#### Acceptance Criteria

1. THE Runtime Monitor SHALL measure per-token latency, TTFT, generation tok/s, VRAM use, RAM use, and Residency Manager activity during generation.
2. THE Joint Controller SHALL choose the Skin Profile and the residency plan together from the VRAM Budget, the Objective, the Working Set size, and monitored latency.
3. THE Joint Controller SHALL apply changes only at Safe Boundaries.
4. WHEN free VRAM falls below the amount the current configuration needs, THE Joint Controller SHALL first shrink the expert cache outside the Working Set and SHALL then narrow the Skin Profile if the shortfall remains.
5. WHEN the Residency Manager reports that the Working Set does not fit, THE Joint Controller SHALL narrow the Skin Profile so that the Working Set fits.
6. WHEN free VRAM rises and stays above the next wider configuration's need for a hold period defined in design, THE Joint Controller SHALL widen the Skin Profile or grow the expert cache, within the Objective's limit.
7. THE Joint Controller SHALL use hysteresis so that it does not alternate between two configurations on consecutive Safe Boundaries.
8. THE Runtime SHALL NOT allocate GPU memory beyond the VRAM Budget, as measured by the Runtime Monitor.
9. IF an out-of-memory error occurs in the Engine, THEN THE Runtime SHALL record it, preserve the conversation history, narrow the configuration, and attempt to resume the session. IF resume is not possible, THEN THE Runtime SHALL report the failure and its cause to the user and SHALL keep the conversation history available.
10. THE Runtime SHALL log every controller decision with its inputs and the resulting configuration.

### Requirement 8: Chat Interface

**User Story:** As a local user, I want to chat with the model privately on my own machine, so that I can get coding help without a cloud service.

#### Acceptance Criteria

1. THE Runtime SHALL provide a chat interface for one local user and one generation stream.
2. THE Runtime SHALL stream generated tokens to the interface as they are produced.
3. THE Runtime SHALL run fully offline after the Package is on disk, and SHALL NOT send prompts or outputs off the machine.
4. THE chat interface SHALL let the user choose an Objective (Speed, Balanced, Quality) without editing configuration files.
5. THE chat interface SHALL let the user change the VRAM Budget during a session, so that the Live Squeeze demo can be run.
6. WHEN the Joint Controller changes configuration during a conversation, THE chat interface SHALL continue the conversation without requiring the user to resend the message.

### Requirement 9: Live Dashboard

**User Story:** As a demo viewer, I want to see what the runtime is doing while it chats, so that the joint controller is visible and not hidden tuning.

#### Acceptance Criteria

1. THE dashboard SHALL show a live VRAM gauge with the current VRAM Budget marked.
2. THE dashboard SHALL show a resident-experts heatmap indicating each expert's tier (VRAM, RAM, SSD) and Working Set membership.
3. THE dashboard SHALL show the current Skin Profile and Objective.
4. THE dashboard SHALL show generation tok/s and TTFT for the current request.
5. THE dashboard SHALL show controller decisions as they happen, for example "budget 6 GB → Slim, evicted 12 cold experts".
6. THE dashboard SHALL update while generation is in progress, at an interval defined in design.

### Requirement 10: Evaluation Harness (2x2 Experiment)

**User Story:** As the builder, I want a controlled experiment that isolates each mechanism, so that the joint-controller claim can be checked against the evidence.

#### Acceptance Criteria

1. THE harness SHALL run four configurations on the same Package and workload: A Full (no adaptive width, no dynamic residency), B Slim (adaptive width only), C Offloaded (dynamic residency only), and D Elastic (both).
2. FOR EACH configuration, THE harness SHALL record TTFT, generation tok/s, prompt-processing tok/s, peak VRAM, peak system RAM, SSD bytes read for weights, quality score on the fixed evaluation set, and out-of-memory count.
3. THE harness SHALL run each configuration under 4, 6, and 8 GB VRAM Budgets.
4. THE harness SHALL use a fixed evaluation set, fixed prompts, and fixed sampling settings across configurations, and SHALL record them with the results.
5. THE harness SHALL output a performance envelope of quality against latency and memory, not only a ranking.
6. THE harness SHALL report results that do not support the claim with the same prominence as results that do.
7. THE fixed evaluation set SHALL be small, fast, and runnable offline, and SHALL consist of a perplexity measurement plus about 20 coding prompts and about 20 reasoning prompts. Design defines the exact contents.
8. THE harness SHALL also run the named baselines from Requirement 2 (llama.cpp full offload, llama.cpp `--n-cpu-moe`, and Edge0-infer) on the same workload, and SHALL report Edge0 as a reference point on a different checkpoint.

### Requirement 11: Success Criteria

**User Story:** As the builder, I want explicit criteria for "it works", so that the submission makes only claims the evidence supports.

#### Acceptance Criteria

1. One Package SHALL support all three width profiles.
2. The Runtime SHALL run under 4, 6, and 8 GB VRAM Budgets without exceeding the configured budget.
3. Reduced-width inference SHALL measurably decrease computation, meaning active expert parameters per token and FLOPs per token decrease as width narrows, and its quality score SHALL stay at or above a usefulness threshold set from the Baseline Report.
4. Dynamic residency SHALL keep the full expert bank accessible without full GPU residency.
5. Configuration D SHALL show a better quality–latency–memory trade-off than at least one of configurations B or C. IF it does not, THEN the write-up SHALL report that result and SHALL NOT claim the joint controller's advantage.

### Requirement 12: Honest Write-up and Prior Art

**User Story:** As a judge, I want the claims to match the evidence and the prior art, so that I can trust the submission.

#### Acceptance Criteria

1. THE README and demo SHALL cite WiSP as prior art for working-set MoE residency.
2. THE README SHALL NOT claim working-set residency for MoE alone, or width adaptation alone, as novel.
3. THE README SHALL cite Liu et al. (1991) and Denning (1968) and SHALL point to the code that implements each.
4. THE README SHALL cite related work named in the handoff (including Edge0, Krasis, MoSE, HydraLLM, CRISP, D2MoE, MC-MoE, Yao et al., and DATS) and DyMoE (arXiv 2603.19172, to be verified in the Feasibility Round) where relevant.
5. EVERY third-party figure quoted in the README or demo SHALL be re-verified against its primary source and linked.
6. EVERY tier claim made without testing on a real lower-tier GPU SHALL state that tiers were simulated by memory budgets only.
7. THE README SHALL state the novelty claim exactly as narrowed in the Novelty boundary section, or as narrowed further by the Feasibility Report.

### Requirement 13: Backend Portability (Stretch)

**User Story:** As an AMD GPU owner, I want the same Package to run on my card, so that Molt is not NVIDIA-only.

#### Acceptance Criteria

1. THE Runtime SHALL support the primary Engine backend on the RTX 4060 before any second backend work begins.
2. WHERE the Engine provides a Vulkan backend, THE project SHALL test the same Package on the RX 6650 XT and SHALL record which requirements pass and fail.
3. IF AMD support fails, THEN THE project SHALL document the failure and SHALL NOT block the NVIDIA submission on it.

### Requirement 14: DeepSeek V4.1 Flash Closing Proof (Stretch)

**User Story:** As a demo viewer, I want to see the same system attempt a much larger model, so that the scaling direction is visible.

#### Acceptance Criteria

1. THE project SHALL begin V4.1 Flash work only after Requirement 11 criteria 1–4 pass on the small MoE.
2. THE project SHALL re-verify all V4.1 Flash figures before quoting them.
3. IF V4.1 Flash runs, THEN THE demo SHALL present it as a closing beat and not as the main demo.

### Requirement 15: Kiroween Submission Compliance

**User Story:** As the builder, I want the submission to meet the contest rules, so that it is eligible.

#### Acceptance Criteria

1. THE repository SHALL be public on GitHub and SHALL include the `.kiro` folder, with `.kiro` not listed in `.gitignore`.
2. THE project SHALL be built with Kiro during the submission window.
3. THE submission SHALL include a demo video shorter than 3 minutes, following the Impossible Model then Live Squeeze structure.
4. THE submission SHALL be made before December 4, 2026, 2:00 PM PT (December 5, 2026, 06:00 WITA), subject to re-checking the official Devpost rules.
5. THE builder SHALL request Kiroween Kiro credits before December 1, 2026, 12:00 PM PT.
6. THE project SHALL record any difference between the official Devpost rules and the dates in the handoff.

## Resolved Decisions (approved 2026-11-11)

1. **Precision skin:** width-only in v1. Precision skins are stretch only (Req 4.5).
2. **Expert skin:** out of scope in v1 because it conflicts with fixed top-K.
3. **Width selection:** per session only in v1. Per-request selection is an optional later extension (Req 5.4).
4. **Mid-generation adaptation:** allowed at Safe Boundaries between tokens, only in response to memory pressure. Choosing width per token by content is excluded. Design defines Safe Boundary mechanics explicitly.
5. **Quality score:** a small, fixed, fast, offline set of perplexity plus about 20 coding and 20 reasoning prompts, with the same prompts and sampling across all configurations (Req 10.7).
6. **Edits:** Req 7.9 softened to attempt resume, preserve history, and report failure. Req 11.3 defines decreased computation as active expert parameters and FLOPs per token falling with width.

## Resolved Decisions — round 2 (approved 2026-11-11, before design)

7. **Novelty:** claim narrowed to a controller that changes width and residency together during a live session when the VRAM budget changes, steered by a user Objective, with width scheduled as mandatory/optional work (Liu et al.). CRISP added to the novelty boundary. The Feasibility Round reads the full CRISP paper and checks whether it adapts online (Req 1.8).
8. **Engine:** design against an Engine Adapter interface. Default is a llama.cpp fork; fallback is an Edge0-infer fork. WiSP/vLLM is dropped as a base (vLLM needs WSL2 on Windows). The Feasibility Round picks one and records why (Req 1.2).
9. **Width source:** training-free conversion that reorders each expert's neurons by importance on calibration data, CRISP-style. Plan around Qwen3-30B-A3B only until the Feasibility Round verifies any other checkpoint (Req 1.9). MoSE is a fallback only if its checkpoints exist.
10. **Prefetch:** off by default, toggleable, measured. The design relies on narrower slices moving fewer bytes per expert miss (Req 6.5).
11. **SSD tier:** an imposed RAM Budget forces the SSD tier into use on machines whose RAM can hold the whole Package (Req 2.6, Req 3.3).
12. **Baselines:** llama.cpp full offload, llama.cpp `--n-cpu-moe`, and Edge0-infer are named baselines (Req 2.1, Req 10.8).
13. **Defaults accepted:** a VRAM filler process simulates 4/6 GB budgets; per-token code is C++ inside the Engine fork; the controller, harness, and API are Python (FastAPI); the dashboard is a web UI over WebSocket; the controller posts a pending config that the Engine applies at the next Safe Boundary; DyMoE is added to related work, to be verified.
