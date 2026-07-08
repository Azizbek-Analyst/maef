# MAEF Roadmap — toward Memory Alignment

**North star: _Memory Alignment_** — keeping a model's behaviour aligned not just with its prompt and
its training, but with whatever ends up in its **long-term memory**, which is an *untrusted* input.
This repo's current results ([README](README.md)) are the first data point; this roadmap is where it goes next.

Memory Alignment is meant as a direction, not a single task: everything below is either **measuring**
the problem or **defending** against it.

---

## Where we are today

- ✅ Reproducible harness ([`maef_harness.py`](maef_harness.py)): 4 subject models + fixed judge, three
  prompt levels (`safety / normal / dangerous`), A/B (control / test / placebo), repeats + 95% CIs,
  dose-response.
- ✅ 7 risks evaluated (`01`–`07`): sycophancy, preference manipulation, truthfulness, sandbagging,
  alignment faking, identity drift, biorisk refusal — with findings written in and raw CSVs in `results/`.
- ✅ Headline result: **weak models break from memory, strong models break from the prompt**;
  robustness is **risk-specific**; safety refusals held across the board; placebo ≈ 0.

**Known gaps to close first:** OpenAI hit an API quota mid-run (01/07 partially unmeasured), and
Gemini's poor showing may be a *model-tier* artifact (Flash, not Pro).

---

## Horizon 1 — Finish the picture (days, cheap)

| Task | What, concretely | Deliverable |
|---|---|---|
| **Re-run OpenAI** | Top up quota; re-run 01 / 07 / 04; optionally add `reasoning_effort` variants of gpt-5.5 | Complete tables, no gaps |
| **Remove tier confound** | Add **Gemini 3.1 Pro** alongside 3.5 Flash | Answers "tier vs. vendor?" — critical for honest claims |
| **Judge robustness** | Add a **second, independent judge** (different vendor) + human spot-check; report agreement (Cohen's κ) | "We measure the subject, not the judge" |

## Horizon 2 — Deepen the science (weeks)

| Task | What, concretely | Deliverable |
|---|---|---|
| **Strengthen poisoning** | Adversarial / optimized notes; multi-turn accumulation; `memory + prompt` combos; jailbreak-style memory. Deepen lightly-covered risks (biorisk in other domains; alignment-faking in real agents) | Pressure→failure curves; upper-bound risk |
| **Poisoned-memory benchmarks** | Overlay poisoning on standard evals — TruthfulQA (truthfulness), MMLU / GSM8K (sandbagging → accuracy drop), AgentBench / τ-bench (agentic). Metric: **capability degradation from memory** | A "poisoned-memory benchmark suite" with citable numbers |
| **Open-weight models** | Llama, Qwen, DeepSeek, Mistral, Gemma | Breadth + opens a **mechanistic** track (look *inside* to see where memory overrides knowledge) |

## Horizon 3 — Build the artifact & the defense (months)

| Task | What, concretely | Deliverable |
|---|---|---|
| **Agent evaluation framework** | Generalize MAEF: agent loops, tool use, pluggable memory stores, standard metrics (`lift` / `dose_to_cross` / `risky_rate`), pluggable risks | Reusable open-source framework |
| **Memory guardrails** | (1) incoming-memory classifier / sanitizer; (2) provenance & trust-scoring per note; (3) "memory = untrusted input" pattern; (4) detector (Opus already flags poisoned notes sometimes — engineer it). Plus an **eval of the defense itself** | Not just "the problem exists" but "here's how to fix it" |

---

## Recommended sequence

1. **OpenAI re-run + Gemini Pro** — fast, closes gaps and the tier confound, refreshes the writeup.
2. **Strengthen poisoning** on 2–3 weak risks — big depth gain, same code.
3. **One poisoned-memory benchmark** (e.g. GSM8K-sandbagging or TruthfulQA) — a comparable, citable number.
4. **Open models** — breadth + mechanistic follow-up.
5. **Framework** and **guardrails** — the flagship, longer contributions, built on top of the evidence base.

**Why this order:** start cheap and trust-building (gaps, confounds); then maximum scientific depth for
minimum new code (poisoning + benchmarks reuse the harness); finish with the two "big" contributions
(framework, guardrails) that turn an interesting experiment into a **direction worth leading**.

---

## Strategic notes

- **Guardrails + Framework** move this from *"I found a problem"* to *"I built a tool and a fix"* — the
  higher-leverage contribution, and a natural fit for a builder rather than a pure theorist.
- **Open models** are the only path to a *mechanistic* answer to "why does memory override knowledge?"
- Synergy with **[Private Layer](#)** (de-identifying personal data): memory guardrails + anonymization
  are two halves of the same problem — trustworthy handling of what a model remembers about a person.

## Contributing

Feedback, replications, and collaborators welcome — especially on judge design, the tier confound, and
realistic (prompt-injection-written) poisoning. Open an issue or a PR.
