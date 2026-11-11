# ELASTIC — Kiroween handoff

> Source of truth for Molt (codename ELASTIC). Stored verbatim from the planning handoff on 2026-11-11, except the copy-paste starter prompt at the end, which was used to start the Kiro session and is omitted here. Figures in this document are as reported by the cited sources and must be re-verified before use in any pitch.

## How to use this handoff

Open this document inside a new Kiro project and build from it there. For Kiroween 2026 the project must be built with Kiro to count, so this file is the transfer point: everything decided so far is below, and the first tasks for the Kiro session are at the end, including a copy-paste starter prompt.

Read the novelty verdict before writing any pitch copy. ELASTIC is worth exploring, but parts of it already exist. The honest claim is narrower — and stronger — than "nobody has done this."

## Project charter

**ELASTIC — Hardware-Adaptive Modular LLM Inference**
Category: Kiroween 2026, Resurrection.

Problem: Local LLM inference is constrained by heterogeneous GPU compute and memory capacity. A smaller model fits a consumer GPU but sacrifices capability. A larger model can sometimes be forced to fit through CPU or disk offloading, but data movement can make it painfully slow. Manually configuring quantization, offloading, and context limits adds further friction.

Solution: A hardware-adaptive runtime for modular Mixture-of-Experts (MoE) models that jointly adjusts expert computational width and weight residency to the hardware available.

Core promise: One model package, multiple hardware configurations, graceful degradation instead of a hard VRAM limit.

Target users: People running local LLMs on consumer Windows PCs with roughly 6–12 GB of GPU memory — local AI enthusiasts, developers who want private coding assistance, and students who cannot justify an expensive GPU upgrade.

What ELASTIC does not promise: identical quality or identical speed on every machine. The goal is the best capability-versus-performance trade-off each machine can sustain, with a consistently usable experience across tiers: responsive conversation, capable coding help, and managed context, within each machine's memory budget.

The user experience should be simple:

- Download or import an ELASTIC-compatible model package.
- Launch ELASTIC and let it profile the hardware.
- Choose an objective — Speed, Balanced, or Quality.
- Start chatting and watch the runtime adapt its resource allocation.

No manual editing of twenty configuration parameters.

## The two mechanisms and the joint controller

ELASTIC has two independent controls. Keeping them separate matters for both the design and the evaluation.

**Mechanism A — Adaptive model capacity.** Change how much of the model is computationally active. Each expert contains nested computational widths, where the small width is a subset of the medium width, which is a subset of the full width. A hardware profile determines how much of each expert executes: narrow experts on a low profile, wider on a medium profile, full-width on a high profile.

**Mechanism B — Dynamic weight residency.** Change where the model's weights live. GPU VRAM holds hot, currently needed weights. System RAM holds warm, likely-to-be-needed weights. SSD holds the cold, persistent weight store. The runtime observes which experts are needed, loads them when necessary, and manages the GPU cache.

The distinction is essential:

- Capacity adaptation reduces the amount of computation performed.
- Residency adaptation reduces the amount of memory required on the GPU.
- Neither automatically reduces the total size of the model stored on disk.

The product claim is the coordination of the two: a joint resource controller that balances computation, memory, transfer latency, and output quality together. Offloading alone is not the claim, and width adaptation alone is not the claim.

For the first implementation, keep expert top-K fixed. Changing expert width already provides a capacity control. Changing top-K at the same time would make it impossible to tell which mechanism produced an improvement.

A working session runs like this: the hardware profiler detects VRAM, RAM, GPU capabilities, and memory pressure; the capacity controller chooses an expert configuration; the residency manager places expert weights across VRAM, RAM, and SSD; the router selects experts and the runtime retrieves the necessary weight slices; and a resource monitor measures latency, memory pressure, and cache activity, adjusting the configuration at safe execution boundaries. If another application starts consuming GPU memory, ELASTIC reduces its resident expert cache or switches to a narrower mode. When resources free up, it can expand again.

## Skin shedding — how capacity adaptation actually works

"Skin shedding" is the working name for dynamic capacity adaptation: an individual expert sheds parameters, width, or precision dynamically, so a much larger model can run in a much smaller memory and compute budget. In the first implementation the skin is selected per session or per request; per-token skin changes are the longer-term direction, not v1 scope.

An expert has three skins it can shed:

- **Width skin.** Keep only the most important neurons in the expert. Neurons are reordered by importance once, offline, so the prefix of the expert is the useful part and a narrower execution uses that prefix. Cutting an arbitrary slice off an untrained expert does not work.

- **Precision skin.** Drop precision per expert, for example from FP4 to 2-bit to 1-bit, using nested (matryoshka-style) quantization so one stored expert supports multiple precisions.

- **Expert skin.** Drop low-confidence experts for a given token entirely, reducing the number of activated parameters for that token.

A skin ladder for a session could look like this:

| Profile | Precision | Width | Experts | Intended effect |
| --- | --- | ---: | ---: | --- |
| Full | FP4 | 100% | Top-6 | Full quality |
| Medium | 2-bit | 75% | Top-6 | About half the active memory |
| Slim | 1-bit | 50% | Top-6 | About a quarter, room for KV cache |
| Emergency | 1-bit | 25% | Top-6 | Still answers, noticeably dumber |

Reducing the number of active experts (for example to top-2) is a further skin beyond v1, where top-K stays fixed so the width effect can be measured cleanly.

The dynamic part is the point: an easy request sheds a lot of skin, a hard request keeps it on. That is the imprecise-computation idea applied per expert.

Prior results cited in the research so far, to be re-verified in Kiro before use in any pitch:

- CRISP reorders neurons by importance so one stored expert supports multiple execution widths, reporting up to 2.1x throughput at the same memory with 3.6% accuracy loss.
- D2MoE uses matryoshka weight quantization, reporting up to 53% peak memory reduction.
- MC-MoE combines mixed precision with online dynamic pruning, reporting 76.6% model compression at 2.54 bits with 3.8% accuracy loss, then a further 15% reduction in activated parameters dynamically with less than 0.6% additional drop.

## Resurrection core — the old papers

ELASTIC only qualifies as Resurrection if an old idea is the actual core of the system, not a paragraph in the README. Two papers fit the two mechanisms almost exactly, and they should be implemented, not just cited.

### Imprecise Computation — Liu et al., 1991 — for capacity

Every task is split into a mandatory subtask and an optional subtask. The mandatory part is required for an acceptable result and must complete before the deadline. The optional part refines the result. The stated goal is graceful degradation: an approximate result of acceptable quality whenever the system is overloaded.

Mapping to ELASTIC:

- Mandatory: the shared core, attention, and the narrow expert slice. This must always run.
- Optional: the wider expert slice, extra precision, and longer context. This is dropped when VRAM or latency is tight.
- Speed / Balanced / Quality modes are simply how much optional work the scheduler allows.

This gives a real scheduler to implement, not just a framing.

### Working Set Model — Denning, 1968 — for residency

Peter Denning, "The Working Set Model for Program Behavior," Communications of the ACM 11(5), May 1968, pages 323–333. The working set is the set of distinct pages referenced in a recent time interval. A program should only run when its working set is in memory, and a page in the working set must not be evicted. Ignoring this produces thrashing.

Mapping to ELASTIC:

- Pages are experts. Main memory is VRAM. Slower storage is RAM and SSD.
- The working set is the set of experts used in the last window of tokens.
- The residency manager keeps the working set resident in VRAM, prefetches experts likely to enter it, and evicts experts outside it. Thrashing — paging experts in and out without progress — is exactly the failure mode ELASTIC exists to prevent.

### Backup framing

Anytime Algorithms (Dean and Boddy, 1988) — algorithms that can be stopped at any time, return some answer, and improve that answer in a well-behaved way as they get more time — is good supporting framing for the write-up, but it is less concrete for expert width than imprecise computation. Keep it as a citation, not the core.

Slimmable Networks (Yu et al., 2019) is the modern bridge to variable-width experts, but it is too recent and too alive to be the Resurrection story itself.

## Novelty verdict — be honest

This was novelty-tested before committing. The verdict shapes the pitch, so do not soften it in Kiro.

**Working Set + MoE is already done.** WiSP (Nokia, August 2026), "A Working-Set View of Mixture-of-Experts Serving on Extremely Low-Resource Hardware," explicitly views low-resource MoE serving as a working-set problem on the GPU. It is a routing-aware expert pager that plugs into an unmodified serving engine and preserves byte-identical outputs, reporting up to 2.0x decode throughput versus static offload on a 24 GB RTX 3090. Its code is public. A judge who searches for it will find it.

**Imprecise Computation + neural networks is already done in parts.** Yao et al. cast deep neural network workflows as imprecise computations with mandatory and optional parts, but for layers and early exits on smaller DNNs, not MoE expert width. DATS (RTSS 2026) decomposes each LLM request into prefill, a validator-backed mandatory prefix, and an optional suffix — but that is tokens, not experts.

**The joint combination was not found.** No work found so far combines imprecise expert-width adaptation and working-set residency in one controller, for a local, single-user MoE, deliberately trading output quality for latency and memory on consumer hardware.

That difference from WiSP is the pitch: WiSP is byte-identical on purpose and only manages residency, so it still has to page the full expert in. ELASTIC manages residency and capacity together, and accepts imprecise output to get a better trade-off when the GPU cannot fit everything.

Rules for the write-up:

- Cite WiSP as prior art. Do not hide it.
- Never claim "we brought the working set to MoE" as the novel contribution.
- Claim only the joint controller and the quality-as-a-variable design, and be ready to show the experiment that isolates it.

## Existing landscape

Both halves of ELASTIC have serious prior art, and the surrounding tools are mature. ELASTIC should build on existing engines rather than compete with them on their own ground.

Already available as foundations: llama.cpp (portable inference, quantization, GPU offloading, multiple backends), Ollama (accessible local model management), LM Studio (model management, memory estimates, GPU offload and KV-cache controls), and ExLlamaV2 (optimized quantized inference for NVIDIA GPUs). A simple automatic model launcher would add little over these.

Relevant research and products:

- MoSE (2026): variable-width MoE experts, including dynamic width selection at inference time.
- HydraLLM (2026): nested model widths and conversion of pretrained models to elastic form.
- FloE (2025): compressed experts loaded on demand.
- OD-MoE (2025): on-demand distributed expert loading.
- DynamicInfer (2026): dynamic neuron loading, caching, and prefetching.
- Edge0 (released November 4, 2026): SSD-based expert streaming with routing prediction on consumer devices, including a unified inference engine and a 35B model. This is a direct competitor on the residency side.
- Krasis (2026): GPU/RAM residency management for large MoE models on NVIDIA consumer systems, with system RAM holding weights and VRAM holding the active working set. This shows the hybrid memory approach is already a usable product.
- AMD's dynamic expert loading for GPT-OSS 20B on Ryzen AI hardware: reported peak memory falling from 13.7 GB to 2.7 GB while throughput fell from 13.4 to 2.4 tokens per second — a clear illustration of the memory-versus-speed trade-off.

## The wow is a capability unlock, not tok/s

ELASTIC is incremental on paper. Its wow will not come from an algorithm name or a tokens-per-second number. Numbers alone invite the response "so… the performance?" — tok/s is abstract, varies by machine, and WiSP and Edge0 already have larger numbers to quote. Faster but dumber is also not impressive on its own; that is just a smaller model.

The wow has to be a moment a judge can see, translated into what a user can now do:

- Not "18 tok/s," but a capable coding assistant running privately and offline on an 8 GB gaming GPU.
- Not "2x throughput," but the same model and the same download working on both an RTX 4060 and an RX 6650 XT, with no config files and no out-of-memory crash.
- Not "efficient," but a student with a normal PC using a capable local model without a cloud subscription or a high-end GPU.

Use two demo beats, in this order:

**Beat 1 — The Impossible Model.** Load a 30–35B MoE on the RTX 4060 8 GB in a normal setup first. It runs out of memory or crawls below 1 tok/s. Then run the same model and the same prompt in ELASTIC: it chats, with the VRAM meter pinned at the budget and no crash. "35B on an 8 GB gaming GPU" needs no paper to understand.

**Beat 2 — The Live Squeeze.** During a live conversation, drop the VRAM budget from 8 GB to 6 GB to 4 GB, or start another GPU-heavy application. A normal setup would die. ELASTIC narrows the experts, evicts cold experts, and keeps talking. Quality dips slightly and speed stays usable. This is the only beat that makes the joint controller visible; without it, the controller is invisible tuning.

Make the dashboard the star of the demo, not a diagram: a live VRAM gauge, a resident-experts heatmap, the current width/skin indicator, and tok/s, all moving while the conversation continues.

Suggested structure for the under-3-minute demo video:

- 0:00 — Baseline fails on the same model, prompt, and machine.
- 0:30 — ELASTIC runs the same thing successfully.
- 1:00 — Live squeeze: budget drops mid-conversation, conversation survives.
- 2:00 — Quick quality check, showing the output is still useful and not fast gibberish.

Open with the task, not a graph: "Here is a coding task I want to do locally and privately on this PC — watch it actually work." Show the performance numbers afterwards, as the explanation for why it works.

## DeepSeek V4.1 Flash — headline extreme, not the main demo

The stretch goal that motivates skin shedding: running DeepSeek V4.1 Flash on consumer hardware, which is practically impossible in normal form. As discussed, the model is roughly 750B parameters, 40 layers, 384 routed experts with top-6 active (about 16B active per token), with a checkpoint in the hundreds of gigabytes.

It has been run on consumer hardware, very slowly. DeepInfer-Mac ran it on a 24 GB MacBook Air with a mixed low-bit conversion occupying 338 GiB on SSD and about 7.6 GiB resident, generating at 0.21–0.72 tokens per second, with a 3,764-token prompt taking 309 seconds to first token. On an 8 GB Windows GPU it would be harder still, with no unified memory and all weight movement over PCIe from SSD.

A shed-skin V4.1 — importance-ordered widths, 1–2 bit nested quantization, reduced active experts — could plausibly run on 8 GB hardware, but expect constant SSD paging, well under a few tokens per second, and long time to first token. Published low-bit builds discussed so far still assume far more system RAM and SSD space than a typical 8 GB laptop has. Re-verify all V4.1 figures in Kiro before quoting any of them publicly.

Decision: V4.1 is the closing proof, not the product demo.

- Main demo: a 30–35B-class MoE running at real chat speed on 8 GB.
- Closing beat: the same system, with the same skin ladder, scaling to V4.1 Flash — running at all is the achievement.

If V4.1 is the main demo, judges will watch sub-1 tok/s output for three minutes and conclude it is broken, even though running at all is the point.

## Deliberately skipped — CPU-ARC

CPU-ARC (Belcak et al., arXiv 2611.09891, submitted November 6, 2026) turns LLM reasoning traces into CPU-executable agentic reasoning caches. On difficult maths and science tasks, those caches matched frontier LLMs at 16–291x lower invocation cost. The pattern is: try the cache first, and on a miss or error, fall through to the full LLM.

It is complementary to ELASTIC, not a replacement: ELASTIC makes the model cheaper to run, CPU-ARC skips the model entirely on a cache hit for a repeated reasoning template.

Decision: skip CPU-ARC for now. Reasons:

- It only pays off for repeated task families with the same reasoning template and different parameters. Open-ended chat misses constantly.
- Building a cache costs roughly 12–84x the original reasoning, so it needs heavy reuse to be worth it.
- It is brand new (November 2026), so it adds nothing to the Resurrection story and would dilute it.

If time remains after the core works, it could return as one small demo task family, but it is not in the MVP and should not be designed for now.

## Hardware for testing

Two test machines are available, both in the same memory tier:

| GPU | VRAM | Role |
| --- | ---: | --- |
| NVIDIA GeForce RTX 4060 | 8 GB | Primary NVIDIA test, CUDA baseline |
| AMD Radeon RX 6650 XT | 8 GB | AMD compatibility test, Vulkan/backend feasibility |

These cards test cross-vendor portability — CUDA versus the AMD software and kernel ecosystem, where support is less mature — not low-tier versus high-tier scaling, because both have 8 GB.

To simulate tiers, impose GPU allocation budgets of 4 GB, 6 GB, and 8 GB on the same hardware. This simulates memory constraints only. It does not simulate the weaker compute of a genuinely lower-tier device. A real 4 GB GPU or weaker integrated GPU is still needed later to validate compute scaling, and any tier claim made before that test should say so.

Start NVIDIA-first. Cross-vendor support is a worthwhile objective, but it must not become a second hackathon hidden inside the first.

A note on context: the builder's messages use the Asia/Makassar (WITA) timezone. Do not infer residence, country, or contest eligibility from the timezone alone.

## Kiroween rules and deadlines

All times below are as stated in the Kiroween 2026 information reviewed on November 11, 2026. Re-check the official Devpost rules in Kiro before relying on them.

- Submission deadline: December 4, 2026 at 2:00 PM PT, which is December 5, 2026 at 06:00 WITA.
- Prize pool: $100,000, with $30,000 for 1st, $20,000 for 2nd, and $10,000 for 3rd.
- Categories: Resurrection, Frankenstein, Skeleton Crew, and Costume Contest. ELASTIC enters Resurrection.
- Only new projects count, and the project must be built during the submission window, with Kiro.
- A public GitHub repository is required, with the `.kiro` folder included in it.
- A demo video under 3 minutes is required.
- Registered builders can request 4,000 free Kiro credits, but the request must be made by December 1 at 12:00 PM PT, which is December 2 at 04:00 WITA.

Timing context: the Epic MegaJam runs November 12–15, 2026, so the realistic ELASTIC build window is roughly November 16 to December 4, 2026. That is enough for a controller, dashboard, and benchmark on top of an existing model and engine. It is not enough to train a slimmable MoE from scratch.

## MVP scope — in and out

Build the smallest system that proves the joint-controller claim end to end.

In scope:

- Hardware profiler: detect GPU, VRAM, RAM, and available backend.
- Elastic model loader: load one model package that supports multiple widths.
- Capacity controller: select expert width (skin) from the resource budget and the chosen objective.
- Expert residency manager: manage GPU, RAM, and SSD placement using a working-set policy.
- Runtime monitor: measure latency, memory, and cache activity, and adjust at safe boundaries.
- Chat interface: real interaction with the model, one local user, one generation stream.
- Benchmark dashboard: live VRAM, resident experts, active width, tok/s, TTFT, and quality trade-offs.
- Three supported expert widths, with width selected per session or per request.
- One primary inference backend. A second backend is a stretch goal.

Out of scope:

- Training a large LLM from scratch.
- Automatic conversion of arbitrary MoE models.
- Dynamic top-K routing.
- Per-token width changes in the first implementation.
- Multi-GPU expert parallelism.
- Cloud inference.
- A complete inference engine written from scratch. Use an existing engine as the foundation, provided its model format and execution path support the required weight slicing.
- CPU-ARC (see the skipped section above).

Initial configuration: one supported elastic MoE model package, three VRAM budgets (4 GB, 6 GB, 8 GB), and the RTX 4060 as the primary target, with the RX 6650 XT as a compatibility investigation.

## How to prove it works

Run a controlled 2x2 experiment on the same model package and the same evaluation workload, so each mechanism's contribution can be isolated:

| Configuration | Adaptive width | Dynamic residency |
| --- | --- | --- |
| A: Full | No | No |
| B: Slim | Yes | No |
| C: Offloaded | No | Yes |
| D: Elastic | Yes | Yes |

Measure, for each configuration:

- Time to first token (TTFT)
- Generation tokens per second
- Prompt-processing tokens per second
- Peak VRAM and peak system RAM
- SSD traffic from weight transfers
- Output quality on a fixed evaluation set
- Out-of-memory frequency and general reliability

Success criteria:

- One model artifact supports all three width profiles.
- It runs under at least three declared VRAM budgets without exceeding the configured GPU allocation.
- Reduced-width inference demonstrably decreases computation while maintaining useful output quality.
- Dynamic residency keeps the full expert bank accessible without requiring full GPU residency.
- The hybrid (Elastic) configuration shows a better quality-latency-memory trade-off than at least one of the single-mechanism alternatives.

Do not invent numerical thresholds in advance. Set them after the first baseline runs. The first benchmark task is to establish whether the problem is real: how performance changes with model size, how context length affects speed and memory, how much backend choice matters across NVIDIA and AMD, and whether adaptive configuration beats the best static configuration. The output should be a performance envelope — quality against latency and memory cost — not just a leaderboard.

## Open feasibility gates

Do not start by coding the whole runtime. The blocking question is where the elastic model comes from, because an ordinary pretrained MoE does not automatically support nested expert widths. Arbitrarily cutting dimensions from a pretrained MoE does not guarantee useful outputs.

Three possible routes, none confirmed yet:

- **Reproduce a small slimmable MoE.** Full architectural control, but real training and implementation work.
- **Adapt an existing elastic model implementation.** Less foundational work, but availability and integration are uncertain. MoSE's public repository has been described as research code with fuller code still to come, and it is an architecture trained from scratch, not a drop-in capable checkpoint. HydraLLM provides a static-to-elastic conversion pipeline, but its public demo is a small dense model (Gemma 3 270M), which proves nested width but is weak for a chat and coding demo.
- **Convert an existing pretrained MoE.** Most practical model relevance, but conversion is architecture-specific and may need distillation work.

Before committing to a route, verify in Kiro: implementation availability, licence, checkpoint quality, and whether the chosen inference backend can actually execute variable-width experts and dynamic GPU residency. Test skin shedding first on a small MoE — a 30B-class model such as a Qwen3 MoE or similar, subject to the feasibility check — where quality versus width versus precision can be measured in a day. V4.1 Flash comes only after that ladder works.

## First tasks in Kiro

Work in this order:

- Request the Kiroween Kiro credits well before the December 1, 12:00 PM PT request deadline.
- Create the public GitHub repository, initialise the Kiro project, and confirm the `.kiro` folder will be committed.
- Re-check the official Kiroween rules on Devpost and record any difference from the deadlines listed here.
- Run the Model Feasibility Round: find a small but meaningful MoE checkpoint, determine whether its experts can support nested widths through an existing implementation or a manageable conversion, and identify a backend that supports both variable-width execution and dynamic residency.
- Establish baselines: run the chosen model in a standard setup (for example llama.cpp or LM Studio) on the RTX 4060 at 4 GB, 6 GB, and 8 GB budgets, across 2K, 4K, and 8K context, on chat, coding, and reasoning prompts. Record TTFT, generation and prompt-processing tok/s, peak VRAM and RAM, and a quality score.
- Prototype residency alone (working-set expert pager) and capacity alone (width/skin selection) as separate toggles, so the 2x2 experiment can be run honestly.
- Build the joint controller, the runtime monitor, and the dashboard that will be the centre of the demo.
- Test the same package on the RX 6650 XT through a compatible backend, as a stretch goal after the NVIDIA path works.
- Only after the small-MoE ladder works, assess a shed-skin DeepSeek V4.1 Flash run as a closing stretch demo.
- Record the demo video last, using the Impossible Model and Live Squeeze structure above, and keep it under 3 minutes.
