#!/usr/bin/env python
# coding: utf-8

# # CS6140 Final v1 — Divide & Conquer Summarization
# 
# **Improvement over baseline:** MapReduce and Refine pipelines that process the
# full document (no truncation) via chunking, targeting higher coverage and
# semantic fidelity.
# 
# **Infrastructure:** NEU Discovery HPC (A100 80 GB), 4-bit quantized Qwen2.5-7B-Instruct.
# 
# | Pipeline | Strategy |
# |----------|----------|
# | MapReduce | Chunk → summarize each independently → merge partial summaries |
# | Refine | Chunk → iteratively refine a running summary through each chunk |
# 
# **Baseline reference (100 samples, A100, 16K-token truncation):**
# 
# | Metric | Baseline | Target |
# |--------|----------|--------|
# | ROUGE-1 F1 | 0.4945 | > 0.50 |
# | ROUGE-2 F1 | 0.1818 | > 0.20 |
# | ROUGE-L F1 | 0.2136 | > 0.23 |
# | BERTScore F1 | 0.0774 | > 0.12 |
# | Avg gen length | 375 words | 450-550 words |

# In[ ]:


import torch
import time
import json
import gc
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


# In[ ]:


def check_system_info():
    """Display system information for debugging."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("Running on CPU (slower performance)")
    print("=" * 60)

check_system_info()


# In[ ]:


# =============================================================
# Configuration — all hyperparameters in one place
# =============================================================

# --- Model ---
USE_QUANTIZATION = '4bit'

# --- MapReduce hyperparameters ---
CHUNK_SIZE = 3000            # tokens per chunk (leaves room for prompt + generation)
CHUNK_OVERLAP = 300          # token overlap between consecutive chunks (~1 paragraph)
MAP_MAX_NEW_TOKENS = 300     # generation budget per chunk summary
MAP_TEMPERATURE = 0.2        # low temp for factual extraction
REDUCE_MAX_NEW_TOKENS = 800  # longer budget to hit 450-550 word target
REDUCE_TEMPERATURE = 0.3
REDUCE_CONTEXT_LIMIT = 8000  # A100 handles this comfortably in the reduce prompt

# --- Refine hyperparameters ---
REFINE_MAX_NEW_TOKENS = 700
REFINE_TEMPERATURE = 0.3

# --- Experiment ---
START_INDEX = 100
END_INDEX = 200    
# Embed the slice into the tags so different HPC nodes save to different checkpoint files natively!
MR_TAG = f"mapreduce_qwen25_7b_hpc_{START_INDEX}to{END_INDEX}"
RF_TAG = f"refine_qwen25_7b_hpc_{START_INDEX}to{END_INDEX}"

RUN_MAPREDUCE = True
RUN_REFINE = True
RUN_FAITHFULNESS_CHECK = True
FAITHFULNESS_MAX_SAMPLES = None  # None => evaluate all generated summaries

print("Configuration loaded.")
print(f"  Chunks: {CHUNK_SIZE} tokens, {CHUNK_OVERLAP} overlap")
print(f"  MAP:    max_new_tokens={MAP_MAX_NEW_TOKENS}, temp={MAP_TEMPERATURE}")
print(f"  REDUCE: max_new_tokens={REDUCE_MAX_NEW_TOKENS}, temp={REDUCE_TEMPERATURE}, context_limit={REDUCE_CONTEXT_LIMIT}")
print(f"  REFINE: max_new_tokens={REFINE_MAX_NEW_TOKENS}, temp={REFINE_TEMPERATURE}")
print(f"  Samples: from idx {START_INDEX} to {END_INDEX}")


# In[ ]:


def load_model(use_quantization=None):
    """Load Qwen 2.5 7B Instruct with optional quantization."""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading model: {model_name}")
    print(f"Quantization: {use_quantization if use_quantization else 'None (FP16/BF16)'}")
    print("This may take a few minutes on first run...\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }

    if use_quantization == '4bit':
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_quantization == '8bit':
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    print("Model loaded successfully!")
    return model, tokenizer

model, tokenizer = load_model(use_quantization=USE_QUANTIZATION)


# In[ ]:


def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """Generate a response from the model using chat template."""
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    if torch.cuda.is_available():
        model_inputs = model_inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Generation function defined.")


# In[ ]:


dataset = load_dataset("ccdv/govreport-summarization")
print(dataset)
test_data = dataset["test"]
print(f"\nTest split: {len(test_data)} samples")
sample_tokens = len(tokenizer.encode(test_data[0]["report"], add_special_tokens=False))
print(f"Sample 0 doc length: {sample_tokens} tokens, {len(test_data[0]['report'])} chars")


# ## Stage 0 — Chunking (Preprocessing)
# 
# Split each full report into overlapping token windows. Unlike the baseline which
# truncates to 16K tokens, the divide-and-conquer pipeline processes **every token**
# of the source document.
# 
# - `chunk_size=3000` keeps each MAP call focused on a manageable section.
# - `overlap=300` (~1 paragraph) prevents information loss at chunk boundaries.

# In[ ]:


def chunk_report(report, tokenizer, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split a report into overlapping token chunks."""
    tokens = tokenizer.encode(report, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_size - overlap  # slide window forward
    return chunks

# Quick sanity check on first sample
sample_chunks = chunk_report(test_data[0]["report"], tokenizer)
total_tokens = len(tokenizer.encode(test_data[0]["report"], add_special_tokens=False))
print(f"Sample 0: {total_tokens} tokens → {len(sample_chunks)} chunks")
for i, c in enumerate(sample_chunks):
    print(f"  Chunk {i}: {len(tokenizer.encode(c, add_special_tokens=False))} tokens, {len(c)} chars")


# ## Stage 1 — MAP Phase (Chunk Summarization)
# 
# Each chunk is independently summarized. The prompt includes positional context
# (`part X of Y`) so the model knows it is seeing a fragment, reducing
# hallucination of conclusions not present in the chunk.

# In[ ]:


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
        model, tokenizer, prompt,
        max_new_tokens=MAP_MAX_NEW_TOKENS,
        temperature=MAP_TEMPERATURE,
    )


def map_phase(model, tokenizer, report):
    """Run MAP over all chunks of a report. Returns (chunks, partial_summaries)."""
    chunks = chunk_report(report, tokenizer)
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        summary = map_summarize_chunk(model, tokenizer, chunk, i, len(chunks))
        partial_summaries.append(summary)
    return chunks, partial_summaries

print("MAP phase functions defined.")


# ## Stage 2 — REDUCE Phase (Merge Summaries)
# 
# Concatenate partial summaries and synthesize a coherent final summary.
# If combined partials exceed `REDUCE_CONTEXT_LIMIT`, a **hierarchical reduce**
# groups them in triples and merges iteratively until they fit.

# In[ ]:


def reduce_phase(model, tokenizer, partial_summaries, max_new_tokens=REDUCE_MAX_NEW_TOKENS):
    """Merge partial summaries into a single coherent summary."""
    combined = "\n\n".join(
        [f"[Section {i+1}]\n{s}" for i, s in enumerate(partial_summaries)]
    )
    combined_tokens = len(tokenizer.encode(combined, add_special_tokens=False))

    if combined_tokens > REDUCE_CONTEXT_LIMIT:
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
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=REDUCE_TEMPERATURE,
    )


def hierarchical_reduce(model, tokenizer, partial_summaries, max_new_tokens=REDUCE_MAX_NEW_TOKENS):
    """
    When partial summaries are too long for a single reduce call,
    group them in triples and reduce in rounds until they fit.
    """
    current = partial_summaries
    round_num = 0

    while True:
        round_num += 1
        print(f"  Hierarchical reduce round {round_num}, {len(current)} segments...")

        # Group into triples
        groups = [current[i:i+3] for i in range(0, len(current), 3)]
        next_level = []

        for group in groups:
            combined = "\n\n".join(group)
            combined_tokens = len(tokenizer.encode(combined, add_special_tokens=False))

            if combined_tokens <= REDUCE_CONTEXT_LIMIT:
                prompt = (
                    "Merge these partial summaries into one concise summary. "
                    "Remove redundancy and keep key facts.\n\n"
                    f"{combined}\n\nMERGED SUMMARY:"
                )
                merged = generate_response(
                    model, tokenizer, prompt,
                    max_new_tokens=400,
                    temperature=MAP_TEMPERATURE,
                )
                next_level.append(merged)
            else:
                # Group still too big — pass items through individually
                for item in group:
                    next_level.append(item)

        # Check if we can do the final reduce
        final_combined = "\n\n".join(next_level)
        final_tokens = len(tokenizer.encode(final_combined, add_special_tokens=False))

        if final_tokens <= REDUCE_CONTEXT_LIMIT or len(next_level) == 1:
            return reduce_phase(model, tokenizer, next_level, max_new_tokens)

        current = next_level

print("REDUCE phase functions defined.")


# In[ ]:


def mapreduce_summarize(model, tokenizer, report):
    """Complete MapReduce summarization pipeline with timing."""
    t0 = time.time()
    chunks, partial_summaries = map_phase(model, tokenizer, report)
    t_map = time.time() - t0

    t1 = time.time()
    final_summary = reduce_phase(model, tokenizer, partial_summaries)
    t_reduce = time.time() - t1

    return {
        "final_summary": final_summary,
        "partial_summaries": partial_summaries,
        "num_chunks": len(chunks),
        "time_map_s": round(t_map, 2),
        "time_reduce_s": round(t_reduce, 2),
        "time_total_s": round(t_map + t_reduce, 2),
    }

print("MapReduce pipeline defined.")


# ## Refine Variant (Iterative Refinement)
# 
# Instead of independent MAP + merge, the Refine approach passes a **running
# summary** forward through each chunk, iteratively incorporating new information.
# 
# ```
# Chunk 1 → Summary_1
# Chunk 2 + Summary_1 → Summary_2 (refined)
# Chunk 3 + Summary_2 → Summary_3 (refined)
# ...
# Chunk N + Summary_{N-1} → Final Summary
# ```
# 
# **Pros:** Maintains coherence; no separate reduce step.
# **Cons:** Sequential only; later chunks have disproportionate influence.

# In[ ]:


def refine_summarize(model, tokenizer, report):
    """Iterative refine summarization with timing."""
    t0 = time.time()
    chunks = chunk_report(report, tokenizer)

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
                "Keep the summary comprehensive but concise (target 400-600 words).\n\n"
                f"CURRENT SUMMARY:\n{running_summary}\n\n"
                f"NEXT SECTION:\n{chunk}\n\n"
                "UPDATED SUMMARY:"
            )
        running_summary = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=REFINE_MAX_NEW_TOKENS,
            temperature=REFINE_TEMPERATURE,
        )

    elapsed = time.time() - t0
    return {
        "final_summary": running_summary,
        "num_chunks": len(chunks),
        "time_total_s": round(elapsed, 2),
    }

print("Refine pipeline defined.")


# ## Checkpointing
# 
# HPC jobs can be preempted. Save after every sample so we can resume from the
# last completed sample if the job restarts.

# In[ ]:


CHECKPOINT_DIR = Path("results")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(results, tag):
    """Save results list to a JSON checkpoint file."""
    path = CHECKPOINT_DIR / f"{tag}_checkpoint.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

def load_checkpoint(tag):
    """Load results from a checkpoint file, or return empty list."""
    path = CHECKPOINT_DIR / f"{tag}_checkpoint.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded checkpoint: {len(data)} samples from {path}")
        return data
    return []

print("Checkpointing utilities defined.")


# ## Run Experiments
# 
# Process test samples through both pipelines. Each sample is checkpointed
# immediately after completion. Periodic `torch.cuda.empty_cache()` keeps
# VRAM fragmentation under control on long runs.

# In[ ]:


# ======================== MapReduce ========================

if RUN_MAPREDUCE:
    if Path("shuffled_indices.json").exists():
        with open("shuffled_indices.json", "r") as f:
            all_indices = json.load(f)
    else:
        all_indices = list(range(len(test_data)))

    target_indices = all_indices[START_INDEX:END_INDEX]

    mr_results = load_checkpoint(MR_TAG)
    done_ids = {r["id"] for r in mr_results}
    run_indices = [i for i in target_indices if i not in done_ids]

    print(f"MapReduce: {len(mr_results)} already done, {len(run_indices)} remaining in this node's slice\n")

    for pos, i in enumerate(tqdm(run_indices, desc="MapReduce"), start=len(mr_results) + 1):
        sample = test_data[i]
        print(f"\n{'='*60}")
        print(f"MapReduce sample {pos}/{END_INDEX - START_INDEX} (idx {i})")

        mr_out = mapreduce_summarize(model, tokenizer, sample["report"])

        mr_results.append({
            "id": i,
            "report": sample["report"],
            "reference_summary": sample["summary"],
            "generated_summary": mr_out["final_summary"],
            "partial_summaries": mr_out["partial_summaries"],
            "num_chunks": mr_out["num_chunks"],
            "time_map_s": mr_out["time_map_s"],
            "time_reduce_s": mr_out["time_reduce_s"],
            "time_total_s": mr_out["time_total_s"],
        })
        save_checkpoint(mr_results, MR_TAG)

        print(f"  Chunks: {mr_out['num_chunks']} | "
              f"MAP: {mr_out['time_map_s']}s | REDUCE: {mr_out['time_reduce_s']}s")
        print(f"  Generated (first 200 chars): {mr_out['final_summary'][:200]}...")

        if pos % 10 == 0:
            torch.cuda.empty_cache()

    print(f"\nMapReduce complete: {len(mr_results)} samples.")


# In[ ]:


# ======================== Refine ========================

if RUN_REFINE:
    if Path("shuffled_indices.json").exists():
        with open("shuffled_indices.json", "r") as f:
            all_indices = json.load(f)
    else:
        all_indices = list(range(len(test_data)))

    target_indices = all_indices[START_INDEX:END_INDEX]

    rf_results = load_checkpoint(RF_TAG)
    done_ids = {r["id"] for r in rf_results}
    run_indices = [i for i in target_indices if i not in done_ids]

    print(f"Refine: {len(rf_results)} already done, {len(run_indices)} remaining in this node's slice\n")

    for pos, i in enumerate(tqdm(run_indices, desc="Refine"), start=len(rf_results) + 1):
        sample = test_data[i]
        print(f"\n{'='*60}")
        print(f"Refine sample {pos}/{END_INDEX - START_INDEX} (idx {i})")

        rf_out = refine_summarize(model, tokenizer, sample["report"])

        rf_results.append({
            "id": i,
            "report": sample["report"],
            "reference_summary": sample["summary"],
            "generated_summary": rf_out["final_summary"],
            "num_chunks": rf_out["num_chunks"],
            "time_total_s": rf_out["time_total_s"],
        })
        save_checkpoint(rf_results, RF_TAG)

        print(f"  Chunks: {rf_out['num_chunks']} | Time: {rf_out['time_total_s']}s")
        print(f"  Generated (first 200 chars): {rf_out['final_summary'][:200]}...")

        if pos % 10 == 0:
            torch.cuda.empty_cache()

    print(f"\nRefine complete: {len(rf_results)} samples.")


# ## Evaluation
# 
# Same metrics as the baseline for direct comparison: ROUGE-1/2/L, BERTScore,
# and self-checking faithfulness.

# In[ ]:


from rouge_score import rouge_scorer

def compute_rouge(results, label=""):
    """Compute ROUGE-1/2/L F1 scores averaged over all samples."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    all_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for r in results:
        scores = scorer.score(r["reference_summary"], r["generated_summary"])
        for key in all_scores:
            all_scores[key].append(scores[key].fmeasure)

    print(f"{'='*50}")
    print(f"ROUGE SCORES (F1) — {label}")
    print(f"{'='*50}")
    for key, values in all_scores.items():
        print(f"  {key.upper()}: {sum(values)/len(values):.4f}")
    return all_scores

if RUN_MAPREDUCE:
    mr_rouge = compute_rouge(mr_results, "MapReduce")
if RUN_REFINE:
    rf_rouge = compute_rouge(rf_results, "Refine")


# In[ ]:


from bert_score import score as bert_score_fn

def compute_bertscore(results, label=""):
    """Compute BERTScore (rescaled with baseline) averaged over all samples."""
    refs = [r["reference_summary"] for r in results]
    cands = [r["generated_summary"] for r in results]
    P, R, F1 = bert_score_fn(
        cands, refs,
        lang="en",
        rescale_with_baseline=True,
        verbose=True,
    )
    print(f"{'='*50}")
    print(f"BERTScore — {label}")
    print(f"{'='*50}")
    print(f"  Precision: {P.mean().item():.4f}")
    print(f"  Recall:    {R.mean().item():.4f}")
    print(f"  F1:        {F1.mean().item():.4f}")
    return {"precision": P, "recall": R, "f1": F1}

if RUN_MAPREDUCE:
    mr_bert = compute_bertscore(mr_results, "MapReduce")
if RUN_REFINE:
    rf_bert = compute_bertscore(rf_results, "Refine")


# In[ ]:


def truncate_report(report, tokenizer, max_input_tokens=4000):
    """Truncate report for faithfulness verification (not for summarization)."""
    tokens = tokenizer.encode(report, add_special_tokens=False)
    if len(tokens) > max_input_tokens:
        tokens = tokens[:max_input_tokens]
        report = tokenizer.decode(tokens, skip_special_tokens=True)
    return report


def self_check_summary(model, tokenizer, report, summary):
    """Ask the model to verify if the summary is faithful to the source report."""
    truncated_doc = truncate_report(report, tokenizer, max_input_tokens=4000)
    prompt = (
        "Given the following source report and its summary, "
        "evaluate whether the summary is factually faithful to the report.\n\n"
        f"REPORT:\n{truncated_doc}\n\n"
        f"SUMMARY:\n{summary}\n\n"
        "Instructions: Does the summary contain only information that is supported by "
        "the report? Answer with one of: 'Faithful', 'Partially Faithful', or 'Unfaithful'. "
        "Then briefly explain why in 1-2 sentences.\n\n"
        "Verdict:"
    )
    return generate_response(
        model, tokenizer, prompt,
        max_new_tokens=120,
        temperature=0.1,
    ).strip()


if RUN_FAITHFULNESS_CHECK:
    pipelines = []
    if RUN_MAPREDUCE:
        pipelines.append(("MapReduce", mr_results, MR_TAG))
    if RUN_REFINE:
        pipelines.append(("Refine", rf_results, RF_TAG))

    for tag_label, res_list, ckpt_tag in pipelines:
        print(f"\nFaithfulness check — {tag_label}")
        eval_set = (
            res_list if FAITHFULNESS_MAX_SAMPLES is None
            else res_list[:FAITHFULNESS_MAX_SAMPLES]
        )
        for j, r in enumerate(tqdm(eval_set, desc=f"Faith-{tag_label}"), 1):
            if "faithfulness_verdict" not in r:
                r["faithfulness_verdict"] = self_check_summary(
                    model, tokenizer, r["report"], r["generated_summary"]
                )
                print(f"  Sample {j}: {r['faithfulness_verdict'][:80]}")
        # Re-save with verdicts
        save_checkpoint(res_list, ckpt_tag)
    print("\nFaithfulness checks complete.")


# ## Extended Evaluation
# 
# Additional metrics from the pipeline guide:
# 
# - **Coverage score**: ROUGE-1 recall between each source chunk and the final
#   summary — verifies all sections are represented, not just early ones.
# - **Redundancy check**: Fraction of repeated bigrams in the final summary —
#   verifies the REDUCE/Refine step properly deduplicates.
# - **Latency breakdown**: Time per phase (MAP, REDUCE, total).

# In[ ]:


# --- Coverage: How well does the final summary represent each source chunk? ---
def compute_coverage(results, tokenizer, label=""):
    """
    For each sample, chunk the original report and compute ROUGE-1 recall
    between each chunk and the final summary. A high average means the summary
    covers content from all parts of the document, not just early sections.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    per_sample_coverage = []

    for r in results:
        chunks = chunk_report(r["report"], tokenizer)
        chunk_scores = []
        for chunk in chunks:
            s = scorer.score(chunk, r["generated_summary"])
            chunk_scores.append(s["rouge1"].recall)
        avg_cov = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0
        per_sample_coverage.append(avg_cov)

    mean_cov = sum(per_sample_coverage) / len(per_sample_coverage)
    print(f"Coverage (avg chunk recall in final summary) — {label}: {mean_cov:.4f}")
    return per_sample_coverage


# --- Redundancy: fraction of repeated bigrams in the summary ---
def compute_redundancy(results, label=""):
    """
    Count repeated bigrams in each generated summary. A low ratio means
    the merge/refine step successfully deduplicated overlapping content.
    """
    ratios = []
    for r in results:
        words = r["generated_summary"].lower().split()
        bigrams = list(zip(words, words[1:]))
        if len(bigrams) == 0:
            ratios.append(0.0)
            continue
        counts = Counter(bigrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        ratios.append(repeated / len(bigrams))

    mean_red = sum(ratios) / len(ratios)
    print(f"Redundancy (repeated bigram ratio) — {label}: {mean_red:.4f}")
    return ratios


# --- Latency breakdown ---
def print_latency(results, label=""):
    """Print per-phase timing statistics."""
    if "time_map_s" in results[0]:
        maps = [r["time_map_s"] for r in results]
        reduces = [r["time_reduce_s"] for r in results]
        totals = [r["time_total_s"] for r in results]
        print(f"Latency — {label}:")
        print(f"  MAP mean:    {sum(maps)/len(maps):.1f}s")
        print(f"  REDUCE mean: {sum(reduces)/len(reduces):.1f}s")
        print(f"  Total mean:  {sum(totals)/len(totals):.1f}s")
    else:
        totals = [r["time_total_s"] for r in results]
        print(f"Latency — {label}: mean total = {sum(totals)/len(totals):.1f}s")


# Run extended metrics
print()
if RUN_MAPREDUCE:
    mr_coverage = compute_coverage(mr_results, tokenizer, "MapReduce")
    mr_redundancy = compute_redundancy(mr_results, "MapReduce")
    print_latency(mr_results, "MapReduce")
    print()

if RUN_REFINE:
    rf_coverage = compute_coverage(rf_results, tokenizer, "Refine")
    rf_redundancy = compute_redundancy(rf_results, "Refine")
    print_latency(rf_results, "Refine")


# ## Three-Way Comparison: Baseline vs MapReduce vs Refine

# In[ ]:


import pandas as pd

def build_comparison():
    """Build a side-by-side comparison table: Baseline vs MapReduce vs Refine."""
    # Load baseline metrics
    baseline_path = Path("results/baseline_qwen25_7b_a100_100samples_faithful_metrics_summary.json")
    rows = []

    if baseline_path.exists():
        with open(baseline_path) as f:
            bl = json.load(f)
        rows.append({
            "Method": "Baseline (single-pass)",
            "ROUGE-1": f"{bl.get('rouge1_f1_mean', 0):.4f}",
            "ROUGE-2": f"{bl.get('rouge2_f1_mean', 0):.4f}",
            "ROUGE-L": f"{bl.get('rougeL_f1_mean', 0):.4f}",
            "BERTScore F1": f"{bl.get('bertscore_f1_mean', 0):.4f}",
            "Coverage": "—",
            "Redundancy": "—",
            "Avg Words": f"{bl.get('gen_length_mean', 0):.0f}",
        })
    else:
        print("WARNING: baseline metrics summary not found, skipping baseline row.")

    def _add_row(name, rouge, bert, cov, red, res_list):
        rows.append({
            "Method": name,
            "ROUGE-1": f"{sum(rouge['rouge1'])/len(rouge['rouge1']):.4f}",
            "ROUGE-2": f"{sum(rouge['rouge2'])/len(rouge['rouge2']):.4f}",
            "ROUGE-L": f"{sum(rouge['rougeL'])/len(rouge['rougeL']):.4f}",
            "BERTScore F1": f"{bert['f1'].mean().item():.4f}",
            "Coverage": f"{sum(cov)/len(cov):.4f}",
            "Redundancy": f"{sum(red)/len(red):.4f}",
            "Avg Words": f"{sum(len(r['generated_summary'].split()) for r in res_list)/len(res_list):.0f}",
        })

    if RUN_MAPREDUCE:
        _add_row("MapReduce", mr_rouge, mr_bert, mr_coverage, mr_redundancy, mr_results)
    if RUN_REFINE:
        _add_row("Refine", rf_rouge, rf_bert, rf_coverage, rf_redundancy, rf_results)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("COMPARISON TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    return df

df_compare = build_comparison()


# ## Save Results
# 
# Persist all outputs: per-sample metrics CSV, full results JSON/JSONL,
# metrics summary JSON, and the comparison table.

# In[ ]:


def full_evaluation_report(results, label):
    """Build per-sample metrics DataFrame and print averages."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for r in results:
        s = scorer.score(r["reference_summary"], r["generated_summary"])
        rows.append({
            "sample_id": r["id"],
            "rouge1_f1": round(s["rouge1"].fmeasure, 4),
            "rouge2_f1": round(s["rouge2"].fmeasure, 4),
            "rougeL_f1": round(s["rougeL"].fmeasure, 4),
            "faithfulness": r.get("faithfulness_verdict", "N/A")[:50],
            "gen_length": len(r["generated_summary"].split()),
            "ref_length": len(r["reference_summary"].split()),
        })
    df = pd.DataFrame(rows)
    print(f"\n--- {label} Averages ---")
    print(df[["rouge1_f1", "rouge2_f1", "rougeL_f1", "gen_length", "ref_length"]].mean())
    return df


OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

save_pairs = []
if RUN_MAPREDUCE:
    save_pairs.append((MR_TAG, mr_results, mr_rouge, mr_bert))
if RUN_REFINE:
    save_pairs.append((RF_TAG, rf_results, rf_rouge, rf_bert))

for tag, res, rouge, bert in save_pairs:
    df = full_evaluation_report(res, tag)

    # JSONL
    with open(OUTPUT_DIR / f"{tag}_results.jsonl", "w", encoding="utf-8") as f:
        for row in res:
            # Exclude bulky fields from JSONL to keep file manageable
            slim = {k: v for k, v in row.items() if k != "report"}
            f.write(json.dumps(slim) + "\n")

    # Full JSON (includes reports for reproducibility)
    with open(OUTPUT_DIR / f"{tag}_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    # Per-sample CSV
    df.to_csv(OUTPUT_DIR / f"{tag}_per_sample_metrics.csv", index=False)

    # Summary JSON
    summary = {
        "run_tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(res),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "rouge1_f1_mean": float(df["rouge1_f1"].mean()),
        "rouge2_f1_mean": float(df["rouge2_f1"].mean()),
        "rougeL_f1_mean": float(df["rougeL_f1"].mean()),
        "gen_length_mean": float(df["gen_length"].mean()),
        "ref_length_mean": float(df["ref_length"].mean()),
        "bertscore_precision_mean": float(bert["precision"].mean().item()),
        "bertscore_recall_mean": float(bert["recall"].mean().item()),
        "bertscore_f1_mean": float(bert["f1"].mean().item()),
    }
    with open(OUTPUT_DIR / f"{tag}_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {tag}")

# Save comparison table
df_compare.to_csv(OUTPUT_DIR / "v1_comparison.csv", index=False)
print("\nAll results saved to results/")


# In[ ]:


# Uncomment to free GPU memory after the run
# del model, tokenizer
# gc.collect()
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     print("GPU memory cleared")

