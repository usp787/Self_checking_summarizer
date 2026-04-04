# CS6140 Final Project — Divide-and-Conquer Direction for Long-Document Summarization

## Goal of this note

This note summarizes the current baseline behavior in `CS6140_final_baseline.ipynb` and proposes a **compatible next-stage pipeline** built around a **divide-and-conquer / map-reduce** idea for AI-agent-style summarization.

The goal is not to replace the current baseline completely, but to evolve it into a stronger system for **long-document summarization** on **GovReport**, while keeping the design internally consistent with the notebook's current choices.

---

## 1. What the current baseline is doing

From the recognizable notebook outputs and code:

- Model: **Qwen/Qwen2.5-7B-Instruct**
- Quantization: **4-bit**
- Input handling: each report is **truncated to 3500 tokens**
- Generation: one-shot summary with **max_new_tokens=400**
- Evaluation subset: **5 samples**
- Metrics shown:
  - **ROUGE-1 F1:** 0.4051
  - **ROUGE-2 F1:** 0.1119
  - **ROUGE-L F1:** 0.1711
- Length statistics:
  - **Average generated length:** 294.4 words
  - **Average reference length:** 574.6 words
- Self-checking:
  - the same model is asked to judge faithfulness
  - source is again truncated, this time to **2500 tokens**
  - visible verdicts are all **Faithful**
- Visible BERTScore output:
  - Precision: 0.0600
  - Recall: 0.0924
  - F1: 0.0775

---

## 2. Big-picture diagnosis of current performance

## 2.1 The baseline is not failing completely

The visible generated summaries are on-topic and coherent. They are not obvious nonsense, and the self-check labels them as faithful in the visible cases. That means the model is already producing a **reasonable first-pass summary** of the material it sees.

So the baseline proves an important point:

> A 4-bit 7B instruct model on Colab T4 can generate plausible summaries for long government reports.

This is a useful baseline.

---

## 2.2 The bigger issue is under-coverage, not pure hallucination

The strongest signal is the **length gap**:

- generated summary ≈ **294 words**
- reference summary ≈ **575 words**

So the current summaries are only about **half as long** as the references.

That strongly suggests the current system is **under-covering the source**, especially because the report is first **truncated to 3500 tokens**. In other words, the system is likely summarizing only the front portion of a very long report, then stopping early.

This interpretation matches the ROUGE pattern:

- **ROUGE-1 is moderate** → the model captures some relevant content
- **ROUGE-2 is low** → phrase-level overlap is weak
- **ROUGE-L is low** → overall coverage and structural overlap are weak

So the current pipeline is best described as:

> a coherent but incomplete baseline that captures high-level content, while missing substantial report coverage.

---

## 2.3 BERTScore should not be over-interpreted yet

The visible BERTScore is extremely low. That may reflect real weakness, but in the current notebook it should be treated cautiously because:

- evaluation is only on **5 samples**
- the summaries are relatively long and structurally different from references
- `rescale_with_baseline=True` changes the scale
- the current pipeline already shows some semantic alignment in visible outputs and ROUGE

So for now:

> treat ROUGE + qualitative inspection as the more stable signal, and treat BERTScore as something to re-check after the pipeline is improved and evaluation size is larger.

---

## 3. Why divide-and-conquer is a strong direction

Your planned direction—using **map-reduce** or a similar divide-and-conquer design—is highly compatible with the current failure mode.

The baseline's largest bottleneck is not necessarily model intelligence. It is that the model cannot reliably summarize the **entire long report** when the pipeline only gives it the first chunk.

This is exactly where divide-and-conquer helps.

### Intuition

A long government report usually contains:

- background
- findings
- evidence
- agency actions
- recommendations
- limitations / implementation issues

A single truncated prompt often over-focuses on the beginning of the report and under-represents later sections.

A divide-and-conquer agent instead does:

1. **divide** the long report into manageable chunks or sections
2. **conquer** each chunk by extracting local summaries or structured notes
3. **reduce** those local summaries into a global summary
4. optionally **refine / verify** the final output

This is a much better fit for long-document summarization than one-shot summarization.

---

## 4. Recommended future pipeline (compatible with your project goal)

Below is a pipeline that is consistent with your current notebook and your future research direction.

## Stage A — Preprocess and segment the report

### Recommended behavior
- Clean obvious formatting noise if needed
- Split the report into **semantic chunks**
  - ideally by section headers / paragraphs
  - fallback to token-based chunking with overlap
- Suggested chunk size:
  - **1200–1800 input tokens**
- Suggested overlap:
  - **100–200 tokens**

### Why this is compatible
This directly replaces the current hard truncation of the entire report at 3500 tokens. Instead of throwing away later content, you keep full-document coverage while still respecting model limits.

---

## Stage B — Map step: local chunk summarization

For each chunk, ask the model to produce a **local summary** or **structured notes**.

### Better than a generic prompt
Instead of only saying “summarize this report,” use a chunk-level instruction like:

- what is the chunk mainly about?
- key findings
- evidence / numbers
- actions or recommendations
- anything that must appear in the final summary

### Optional structured intermediate format
Instead of free-form chunk summaries, produce a schema like:

```text
Chunk topic:
Key findings:
Evidence:
Recommendations / actions:
Important entities:
Confidence / uncertainty:
```

This is often better for reduction because the second-stage model gets more organized inputs.

### Why this helps
The map stage turns a long document into a set of **compressed, coverage-preserving intermediate representations**.

---

## Stage C — Reduce step: merge local summaries into a global summary

Feed the chunk summaries into a second prompt that asks for a final integrated summary.

### Important constraint
Tell the reducer to:
- remove duplicates
- preserve the most important findings
- keep recommendations and policy implications
- produce a target length close to the reference range

For GovReport-like data, you may want something like:

- target final summary length: **500–650 words**

### Why this is compatible
Your current baseline is too short. The reducer stage gives you a direct place to enforce final summary length and broader coverage.

---

## Stage D — Optional refine step

After the reduce step, run one refinement pass:

- improve coherence
- remove repetition
- compress weakly important details
- ensure the final summary reads like one document rather than stitched chunks

This should be a light editing step, not a full rewrite.

---

## Stage E — Verification / critic step

If you want an agentic flavor, add a separate verifier that checks:

- unsupported claims
- missing critical findings
- duplication
- mismatch between recommendations and findings

But this verifier should ideally operate on:
- the **intermediate chunk notes**, and/or
- the **full set of chunk summaries**,
not only a short truncated prefix.

This is more trustworthy than the current self-check, which uses another truncated view.

---

## 5. Suggested agent framing for your project

If you want to explicitly present this as an **AI agent** rather than only a summarization script, the following framing is clean and compatible:

### Agent roles

#### 1. Planner / Segmenter
- decides how to split the document
- preserves section boundaries when possible

#### 2. Mapper
- summarizes each chunk into structured notes

#### 3. Reducer
- combines chunk notes into a global summary

#### 4. Critic / Verifier
- checks factual support, missing items, repetition, and length balance

#### 5. Refiner
- polishes the final answer into a clean human-readable summary

This gives you a strong “divide and conquer” narrative:
the system solves a hard long-context task by breaking it into smaller reliable subtasks.

---

## 6. Concrete hypotheses you can test in the project

This direction is strongest when it is framed as a comparison against the current baseline.

## Baseline hypothesis
> A one-shot summarization pipeline on truncated long reports produces coherent but incomplete summaries.

## Main project hypothesis
> A divide-and-conquer map-reduce summarization pipeline improves coverage and overall summary quality over one-shot truncation-based summarization.

## Optional secondary hypotheses
1. Structured chunk notes outperform free-form chunk summaries in the reduce step.
2. Section-aware chunking outperforms naive fixed-window chunking.
3. A critic/refiner stage improves faithfulness or reduces redundancy.
4. Deterministic decoding produces more stable evaluation results than sampled decoding.

These hypotheses are clean, testable, and directly aligned with your notebook.

---

## 7. Priority improvements to the pipeline

If you want the roadmap in practical order, use this order:

## Priority 1 — Replace hard truncation with chunked map-reduce
This is the single most important upgrade.

## Priority 2 — Make target final length explicit
The current summaries are too short relative to references.

## Priority 3 — Use more structured prompts
Especially at the chunk level.

## Priority 4 — Make evaluation more reliable
Increase from 5 samples to something more convincing, such as:
- 50 samples for pilot experiments
- 100+ if runtime permits

## Priority 5 — Improve the verification stage
Avoid relying only on the same model judging a truncated view.

---

## 8. Practical pipeline advice so the whole system stays compatible

Below are implementation choices that work well together.

## 8.1 Keep the current backbone first
You do **not** need to switch models immediately.

A good experimental strategy is:

- keep **Qwen2.5-7B-Instruct**
- improve the pipeline first
- compare one-shot baseline vs divide-and-conquer using the same backbone

This isolates the effect of the pipeline.

---

## 8.2 Use deterministic decoding for evaluation
In the current notebook, generation uses sampling settings such as low temperature.

For benchmark evaluation, deterministic decoding is usually better:

- `do_sample=False`

This reduces variance and makes comparisons fairer.

---

## 8.3 Use structured intermediate outputs
This is especially important in map-reduce systems.

Why?

Because free-form chunk summaries may drift in style and omit key dimensions. Structured notes make the reduce step more controllable.

---

## 8.4 Use separate prompts for map and reduce
Do not use the exact same summarization instruction for every stage.

### Map prompt should optimize for:
- local coverage
- extraction of key points
- minimal hallucination

### Reduce prompt should optimize for:
- global integration
- de-duplication
- coherent narrative
- target final length

This separation is central to compatibility.

---

## 8.5 Evaluate both quality and coverage
Your current notebook already computes lengths. Keep that idea.

Track at least:

- ROUGE-1 / ROUGE-2 / ROUGE-L
- average generated length
- average reference length
- length ratio
- latency per document
- optional faithfulness flag / judge result

This is useful because divide-and-conquer systems may improve **coverage** even before they dramatically improve every semantic metric.

---

## 9. Suggested experiment table for the project

A clean experiment plan could be:

### Exp 1 — Current baseline
- one-shot
- first 3500 tokens only
- direct summary

### Exp 2 — One-shot with stronger prompt and length control
- same truncation
- better prompt
- explicit 500–650 word target

### Exp 3 — Map-reduce with fixed-size chunking
- chunk report
- local summaries
- final merge

### Exp 4 — Map-reduce with structured chunk notes
- same as Exp 3
- but mapper outputs structured notes instead of free-form summary

### Exp 5 — Map-reduce + critic/refiner
- add verification and cleanup stage

This gives a coherent ablation ladder.

---

## 10. Colab T4 inference tips for this project

Your notebook feels slow even on 5 examples because it is doing more work than it first appears:

- long input prompts (~3500 tokens)
- long output generation (~400 tokens)
- additional self-check generations
- metric computation including BERTScore, which loads a separate model

For Colab T4, the following advice is compatible with the future pipeline.

## 10.1 Use FP16 compute instead of BF16 in 4-bit mode on T4
In the notebook, 4-bit compute dtype is set to `torch.bfloat16`.

For T4, **FP16 is usually the safer/faster choice**.

So change:

```python
bnb_4bit_compute_dtype=torch.bfloat16
```

to:

```python
bnb_4bit_compute_dtype=torch.float16
```

---

## 10.2 Use `model.eval()` and inference mode
For inference-only experiments:

```python
model.eval()
with torch.inference_mode():
    ...
```

This is a small but useful improvement.

---

## 10.3 Do not run expensive evaluation in the inner loop
During development:

- generate summaries first
- use ROUGE on a small subset
- postpone BERTScore until the pipeline is stable
- run self-check only on selected examples

This greatly speeds iteration.

---

## 10.4 Lower token budgets during debugging
For rapid iteration, use smaller settings first.

For example:
- smaller chunk count
- shorter chunk summaries
- lower `max_new_tokens`

Then scale up after the pipeline logic works.

---

## 10.5 Map-reduce can also help runtime control
Even though map-reduce adds more steps, it may still be operationally easier because:
- you can cache chunk-level results
- you can rerun only the reduce step after prompt changes
- you can inspect which stage causes errors
- you avoid wasting effort on a single massive prompt that covers only the front of the report

So divide-and-conquer is not only a quality idea; it is also a workflow and debugging advantage.

---

## 11. Recommended final project positioning

A strong way to present the project is:

> We start from a one-shot summarization baseline that truncates long government reports and produces coherent but incomplete summaries. We then propose a divide-and-conquer AI-agent pipeline inspired by map-reduce: the agent segments the document, summarizes each part, integrates local evidence into a global summary, and optionally verifies the final output. This design is intended to improve full-document coverage, summary quality, and controllability under limited inference resources.

This framing is compatible with:
- your current notebook
- your “conquer and divide” motivation
- long-document summarization
- future agentic extensions

---

## 12. Final recommendation

The most important idea is simple:

> Do not spend your next effort only on prompt polishing for the current one-shot truncation pipeline.

Instead, move toward a **divide-and-conquer summarization agent** as soon as possible.

### Best next step
Build a **minimal map-reduce baseline** first:

1. chunk the report
2. summarize each chunk
3. merge chunk summaries
4. compare against the current one-shot baseline

If that works, then add:
- structured chunk notes
- critic/refiner stage
- stronger evaluation

That path is both scientifically clean and highly compatible with your project goal.
