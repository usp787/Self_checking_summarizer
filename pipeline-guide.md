# From Claude
# CS6140 Final Project — Divide & Conquer Summarization Pipeline Guide

## 1. Motivation

The baseline notebook (Qwen2.5-7B-Instruct, 4-bit, T4 GPU) reveals two core problems:

| Metric | Baseline Value | Root Cause |
|--------|---------------|------------|
| ROUGE-1 F1 | 0.405 | Input truncated to 3,500 tokens; model sees < 25% of avg report |
| ROUGE-2 F1 | 0.112 | Low phrase-level overlap due to missing context |
| ROUGE-L F1 | 0.171 | Short generations (~294 words) vs references (~575 words) |
| BERTScore F1 | 0.078 | Semantic coverage is marginal |

[(GovReport)](https://huggingface.co/datasets/ccdv/govreport-summarization) documents average ~15,000+ tokens. Feeding the full document into a single LLM call is infeasible on a T4 with a 7B model. **Divide-and-conquer** solves this by decomposing the problem into smaller, GPU-friendly sub-tasks.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FULL REPORT (15K+ tokens)             │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   CHUNK / SPLIT │  (Stage 0: Preprocessing)
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ Chunk 1  │  │ Chunk 2  │  │ Chunk N  │
   │ ≤ 3K tok │  │ ≤ 3K tok │  │ ≤ 3K tok │
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │              │              │
        ▼              ▼              ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │  MAP:    │  │  MAP:    │  │  MAP:    │   (Stage 1: Map)
   │ Summarize│  │ Summarize│  │ Summarize│
   │ chunk    │  │ chunk    │  │ chunk    │
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
              ┌────────▼────────┐
              │    REDUCE:      │  (Stage 2: Reduce)
              │ Merge partial   │
              │ summaries into  │
              │ final summary   │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  FINAL SUMMARY  │
              └─────────────────┘
```

---

## 3. Stage-by-Stage Implementation Plan

### Stage 0 — Chunking (Preprocessing)

Split each report into overlapping chunks so no context is lost at boundaries.

```python
def chunk_report(report, tokenizer, chunk_size=3000, overlap=300):
    """
    Split report into overlapping token chunks.
    
    Args:
        report: raw report text
        tokenizer: HF tokenizer
        chunk_size: max tokens per chunk (leave room for prompt + generation)
        overlap: token overlap between consecutive chunks
    
    Returns:
        list of text chunks
    """
    tokens = tokenizer.encode(report, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += chunk_size - overlap  # slide window forward
    return chunks
```

**Design decisions:**

- `chunk_size=3000` keeps each MAP call well within the 4-bit model's comfortable range on T4, leaving room for the prompt template (~200 tokens) and generation (~300 tokens).
- `overlap=300` (~1 paragraph) prevents information from being split across chunk boundaries.
- For structured reports (numbered sections, headings), consider **section-aware splitting** instead — detect heading patterns via regex and split at section boundaries.

### Stage 1 — MAP (Parallel Chunk Summarization)

Each chunk gets independently summarized with a focused prompt.

```python
def map_summarize_chunk(model, tokenizer, chunk, chunk_index, total_chunks):
    """Summarize a single chunk with positional context."""
    prompt = (
        f"You are summarizing part {chunk_index + 1} of {total_chunks} "
        f"of a government report. Extract the key findings, data points, "
        f"and policy recommendations from this section. Be specific and "
        f"preserve important details.\n\n"
        f"SECTION:\n{chunk}\n\n"
        f"SECTION SUMMARY:"
    )
    return generate_response(
        model, tokenizer,
        prompt,
        max_new_tokens=300,
        temperature=0.2  # low temp for factual extraction
    )


def map_phase(model, tokenizer, report):
    """Run MAP over all chunks of a report."""
    chunks = chunk_report(report, tokenizer)
    print(f"  Split into {len(chunks)} chunks")
    
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  MAP chunk {i+1}/{len(chunks)}...")
        summary = map_summarize_chunk(model, tokenizer, chunk, i, len(chunks))
        partial_summaries.append(summary)
    
    return partial_summaries
```

**Tip — Positional context in the prompt** (`part X of Y`) helps the model understand it's seeing a fragment, not the whole document. This reduces hallucination of conclusions that aren't in the chunk.

### Stage 2 — REDUCE (Merge Summaries)

Concatenate partial summaries and ask the model to synthesize a coherent final summary.

```python
def reduce_phase(model, tokenizer, partial_summaries, max_new_tokens=600):
    """Merge partial summaries into a single coherent summary."""
    combined = "\n\n".join(
        [f"[Section {i+1}]\n{s}" for i, s in enumerate(partial_summaries)]
    )
    
    # Check if combined text fits in one reduce call
    combined_tokens = len(tokenizer.encode(combined, add_special_tokens=False))
    
    if combined_tokens > 3500:
        # Recursive reduce: re-chunk and summarize again
        print(f"  Combined partials too long ({combined_tokens} tokens), "
              f"running hierarchical reduce...")
        return hierarchical_reduce(model, tokenizer, partial_summaries, max_new_tokens)
    
    prompt = (
        "Below are section-by-section summaries of a government report. "
        "Synthesize them into a single, coherent, and comprehensive summary. "
        "Eliminate redundancy, maintain logical flow, and preserve all key "
        "findings, data, and recommendations. "
        "Target length: 400-600 words.\n\n"
        f"{combined}\n\n"
        "FINAL SUMMARY:"
    )
    
    return generate_response(
        model, tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.3
    )


def hierarchical_reduce(model, tokenizer, partial_summaries, max_new_tokens=600):
    """
    When partial summaries are too long for a single reduce call,
    group them and reduce in rounds until they fit.
    """
    current = partial_summaries
    round_num = 0
    
    while True:
        round_num += 1
        print(f"  Reduce round {round_num}, {len(current)} segments...")
        
        # Group into pairs/triples
        groups = [current[i:i+3] for i in range(0, len(current), 3)]
        next_level = []
        
        for g, group in enumerate(groups):
            combined = "\n\n".join(group)
            combined_tokens = len(tokenizer.encode(combined, add_special_tokens=False))
            
            if combined_tokens <= 3500:
                prompt = (
                    "Merge these partial summaries into one concise summary. "
                    "Remove redundancy and keep key facts.\n\n"
                    f"{combined}\n\nMERGED SUMMARY:"
                )
                merged = generate_response(
                    model, tokenizer, prompt,
                    max_new_tokens=400, temperature=0.2
                )
                next_level.append(merged)
            else:
                # If even a single group is too big, summarize individually
                for item in group:
                    next_level.append(item)
        
        # Check if we can do final reduce
        final_combined = "\n\n".join(next_level)
        final_tokens = len(tokenizer.encode(final_combined, add_special_tokens=False))
        
        if final_tokens <= 3500 or len(next_level) == 1:
            return reduce_phase(model, tokenizer, next_level, max_new_tokens)
        
        current = next_level
```

### Full Pipeline

```python
def mapreduce_summarize(model, tokenizer, report, max_new_tokens=600):
    """Complete MapReduce summarization pipeline."""
    print("Phase 1: MAP — chunking and summarizing sections...")
    partial_summaries = map_phase(model, tokenizer, report)
    
    print(f"\nPhase 2: REDUCE — merging {len(partial_summaries)} partial summaries...")
    final_summary = reduce_phase(model, tokenizer, partial_summaries, max_new_tokens)
    
    return final_summary
```

---

## 4. Evaluation Strategy

Keep the same metrics from the baseline for direct comparison.

| Metric | What It Measures | Baseline | Target |
|--------|-----------------|----------|--------|
| ROUGE-1 F1 | Unigram overlap | 0.405 | > 0.45 |
| ROUGE-2 F1 | Bigram overlap | 0.112 | > 0.15 |
| ROUGE-L F1 | Longest common subsequence | 0.171 | > 0.22 |
| BERTScore F1 | Semantic similarity | 0.078 | > 0.15 |
| Faithfulness (self-check) | Factual consistency | 5/5 | Maintain |
| Avg generation length | Coverage proxy | 294 words | 450-550 words |

**Additional evaluation for the divide-and-conquer approach:**

- **Coverage score**: Compute ROUGE between each chunk's content and the final summary to verify all sections are represented, not just early ones.
- **Redundancy check**: Count repeated n-grams in the final summary to verify the REDUCE step properly deduplicates.
- **Latency breakdown**: Time each phase (chunking, MAP total, REDUCE total) to understand the cost of the multi-pass approach.

---

## 5. Variants to Explore

### 5.1 MapReduce (baseline divide-and-conquer)

As described above. Simplest to implement, good first step.

### 5.2 Map-Refine (iterative refinement)

Instead of summarizing chunks independently and merging, **accumulate context** by passing the running summary forward:

```
Chunk 1 → Summary_1
Chunk 2 + Summary_1 → Summary_2 (refined)
Chunk 3 + Summary_2 → Summary_3 (refined)
...
Chunk N + Summary_{N-1} → Final Summary
```

**Pros:** Maintains coherence across the document; no separate reduce step.  
**Cons:** Sequential only (no parallelism); later chunks have disproportionate influence; errors compound.

```python
def refine_summarize(model, tokenizer, report, max_new_tokens=600):
    """Iterative refine summarization."""
    chunks = chunk_report(report, tokenizer, chunk_size=3000, overlap=300)
    
    running_summary = ""
    for i, chunk in enumerate(chunks):
        if i == 0:
            prompt = (
                "Summarize the following section of a government report. "
                "Focus on key findings and recommendations.\n\n"
                f"SECTION:\n{chunk}\n\nSUMMARY:"
            )
        else:
            prompt = (
                "Below is your running summary of a government report so far, "
                "followed by the next section. Update and refine the summary "
                "to incorporate new information from this section. "
                "Keep the summary comprehensive but concise.\n\n"
                f"CURRENT SUMMARY:\n{running_summary}\n\n"
                f"NEXT SECTION:\n{chunk}\n\n"
                "UPDATED SUMMARY:"
            )
        
        running_summary = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.3
        )
        print(f"  Refined through chunk {i+1}/{len(chunks)}")
    
    return running_summary
```

### 5.3 Hybrid Map-Cluster-Reduce

Group semantically similar chunks before reducing, so the reduce step merges thematically coherent groups rather than arbitrary positional groups.

```
Chunks → Embed each chunk → Cluster (k-means) → Reduce per cluster → Final reduce
```

This is more complex but handles reports with interleaved topics well.

---

## 6. T4 Colab Speed Optimization

### 6.1 Immediate Wins

| Optimization | Expected Speedup | Implementation |
|-------------|-----------------|----------------|
| Verify KV-cache is enabled | ~2x if it was off | `model.config.use_cache = True` (usually default) |
| Reduce `max_new_tokens` in MAP phase | proportional | 200-250 is enough for chunk summaries |
| Use greedy decoding in MAP | ~10-15% | `do_sample=False, temperature=None` for factual extraction |
| Batch-decode cleanup | minor | Already doing this correctly |

### 6.2 Model Selection

For a T4 with 16GB VRAM, consider this trade-off:

| Model | VRAM (4-bit) | Speed (tok/s est.) | Quality |
|-------|-------------|-------------------|---------|
| Qwen2.5-3B-Instruct | ~3 GB | ~40-50 | Good for MAP chunks; weaker REDUCE |
| Qwen2.5-7B-Instruct | ~5 GB | ~20-25 | Current baseline; solid quality |
| Qwen2.5-14B-Instruct | ~10 GB | ~10-12 | Better quality; tight on VRAM |

**Practical suggestion:** Use the 7B model for both phases, but if speed is critical, use 3B for the MAP phase (simpler extraction task) and 7B only for the REDUCE phase (harder synthesis task).

### 6.3 Scaling to More Samples

For your final evaluation (e.g., 50-100 samples):

- Pre-compute all chunks and save them to disk before running inference.
- Run MAP in a loop with periodic `torch.cuda.empty_cache()` calls.
- Save intermediate partial summaries to a JSON file after each sample so you can resume if Colab disconnects.
- Consider running overnight with Colab Pro's longer session limits.

```python
import json

CHECKPOINT_FILE = "mapreduce_results_checkpoint.json"

def save_checkpoint(results, path=CHECKPOINT_FILE):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

def load_checkpoint(path=CHECKPOINT_FILE):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
```

---

## 7. Experiment Plan

### Phase 1 — Validate MapReduce (Week 1)

1. Implement `chunk_report`, `map_phase`, `reduce_phase` as above.
2. Run on same 5 test samples from baseline.
3. Compare ROUGE/BERTScore side-by-side with baseline.
4. Verify faithfulness is maintained.

### Phase 2 — Implement Refine Variant (Week 1-2)

1. Implement `refine_summarize`.
2. Run on same 5 samples.
3. Three-way comparison: Baseline vs MapReduce vs Refine.

### Phase 3 — Scale & Ablate (Week 2)

1. Pick the better method from Phase 2.
2. Scale to 50+ test samples.
3. Ablation studies:
   - Chunk size: 2000 vs 3000 vs 4000 tokens
   - Overlap: 0 vs 150 vs 300 tokens
   - MAP generation length: 150 vs 250 vs 350 tokens
   - REDUCE generation length: 400 vs 600 vs 800 tokens

### Phase 4 — Final Report (Week 3)

1. Full evaluation on 100 test samples with best config.
2. Latency analysis (time per sample breakdown).
3. Qualitative examples (side-by-side comparisons).
4. Write-up connecting to divide-and-conquer / AI agent literature.

---

## 8. File Structure

```
CS6140_final/
├── CS6140_final_baseline.ipynb          # Original baseline (keep as-is)
├── CS6140_final_mapreduce.ipynb         # MapReduce pipeline
├── CS6140_final_refine.ipynb            # Refine variant
├── CS6140_final_evaluation.ipynb        # Side-by-side comparison & plots
├── utils/
│   ├── chunking.py                      # chunk_report, section-aware split
│   ├── generation.py                    # generate_response (shared)
│   ├── evaluation.py                    # ROUGE, BERTScore, faithfulness
│   └── checkpointing.py                # save/load intermediate results
├── results/
│   ├── baseline_results.json
│   ├── mapreduce_results.json
│   └── refine_results.json
├── pipeline_guide.md                    # This document
└── README.md
```

---

## 9. Key References

- **MapReduce for LLM summarization**: LangChain's MapReduce chain is the most well-known implementation of this pattern. The core idea maps directly to the classic distributed computing paradigm.
- **Iterative refinement**: The "refine" chain in LangChain processes documents sequentially, updating a running summary — analogous to a fold/reduce operation.
- **GovReport benchmark**: Huang et al., 2021 — "Efficient Attentions for Long Document Summarization." Provides the dataset and strong baselines using LED and BigBird models.
- **Divide-and-conquer agents**: The pattern generalizes beyond summarization to any task where input exceeds context limits — RAG, multi-document QA, code analysis, etc.
