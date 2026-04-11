#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Generating with a 7B model on a Colab T4 can be painfully slow, especially when processing sequences that are thousands of tokens long. While bitsandbytes 4-bit quantization (NF4) is heavily optimized for memory savings during QLoRA training—which makes it excellent for fine-tuning pipelines—it is notoriously slow for pure inference because the weights have to be continuously dequantized on the fly.

Here is how you can dramatically speed up your inference on a T4:

Switch to AWQ or GPTQ Quantization: Instead of loading the base model and quantizing it dynamically with bitsandbytes, download a pre-quantized model. Look for Qwen/Qwen2.5-7B-Instruct-AWQ on Hugging Face. AWQ (Activation-aware Weight Quantization) is designed specifically for fast inference and will yield a noticeable speed boost over NF4 while maintaining similar quality.

Use vLLM: Standard Hugging Face generate() is not optimized for speed. If you wrap your generation pipeline in a high-throughput serving engine like vLLM (which uses PagedAttention to manage key-value cache memory efficiently), you can often see a 3x to 5x speedup, even on a single T4.

Enable SDPA (Scaled Dot Product Attention): In your model_kwargs, you can add "attn_implementation": "sdpa". The T4 GPU doesn't fully support the hardware-level Flash Attention 2, but PyTorch's native SDPA provides a memory-efficient attention mechanism that will still speed up processing for long context windows.

Implement Batching: You are currently iterating through the dataset one sample at a time. The T4 GPU shines when it can process matrices in parallel. Pass a list of 2 to 4 prompts to the tokenizer and model simultaneously. Keep in mind that for batching to work properly, you will need to set tokenizer.padding_side = "left" and ensure you are passing the attention_mask to the generate() function.
"""


# In[ ]:


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


# In[ ]:


def check_system_info():
    """Display system information for debugging"""
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


# Configuration - Change this based on your hardware
USE_QUANTIZATION = '4bit'  # Options: None, '8bit', '4bit'

print(f"Configuration: Quantization = {USE_QUANTIZATION if USE_QUANTIZATION else 'None (Full Precision)'}")


# In[ ]:


from transformers import BitsAndBytesConfig

# Model kwargs are now defined cleanly inside the load_model() function below.


# In[ ]:


def load_model(use_quantization=None):
    """
    Load Qwen 2.5 7B Instruct model

    Args:
        use_quantization: None, '8bit', or '4bit'
    """
    model_name = "Qwen/Qwen2.5-7B-Instruct"

    print(f"Loading model: {model_name}")
    print(f"Quantization: {use_quantization if use_quantization else 'None (FP16/BF16)'}")
    print("This may take a few minutes on first run...")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Configure model loading based on quantization
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",  # Automatically distribute across available devices
        "attn_implementation": "sdpa"  # try tile attention for vram
    }

    if use_quantization == '8bit':
        print("Loading with 8-bit quantization (requires bitsandbytes)")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif use_quantization == '4bit':
        print("Loading with 4-bit quantization (requires bitsandbytes)")
        # Dynamically switch between bfloat16 (A100) and float16 (V100/T4)
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,      # Dynamically set compute dtype
            bnb_4bit_use_double_quant=True,            # saves a bit more VRAM
            bnb_4bit_quant_type="nf4"                  # NF4 is standard for QLoRA/inference
        )
    else:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
            print("Using BF16 precision")
        else:
            model_kwargs["torch_dtype"] = torch.float16
            print("Using FP16 precision")

    # Load model
    # pip install -U bitsandbytes>=0.46.1, for colab environment
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    print("Model loaded successfully!")

    return model, tokenizer

# Load the model
model, tokenizer = load_model(use_quantization=USE_QUANTIZATION)


# In[ ]:


def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    """Generate a response from the model"""

    # Format the prompt using chat template
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt")

    # Move to same device as model
    if torch.cuda.is_available():
        model_inputs = model_inputs.to(model.device)

    # Generate
    print("Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated tokens (exclude the prompt)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

print("Generation function defined successfully!")


# In[ ]:


"""
# Example 1: Fibonacci sequence
prompt1 = "Introduce the linear regression."
# prompt1 = 'Write a python function to print hello world'
print("=" * 60)
print(f"Prompt: {prompt1}")
print("=" * 60)

response1 = generate_response(
    model,
    tokenizer,
    prompt1,
    max_new_tokens=512,
    temperature=0.7
)

print("\nResponse:")
print(response1)
print("=" * 60)
"""


# In[ ]:


# Install datasets library if not present
# !pip install -q datasets rouge-score bert-score

from datasets import load_dataset

dataset = load_dataset("ccdv/govreport-summarization")
print(dataset)
print("\nSample keys:", dataset["train"][0].keys())
print("Doc length (chars):", len(dataset["train"][0]["report"]))
print("Summary length (chars):", len(dataset["train"][0]["summary"]))


# In[ ]:


def truncate_report(report, tokenizer, max_input_tokens=3500):
    """
    Truncate report to fit within model's practical context window.
    3500 leaves room for the prompt template + generated summary.
    """
    tokens = tokenizer.encode(report, add_special_tokens=False)
    if len(tokens) > max_input_tokens:
        tokens = tokens[:max_input_tokens]
        report = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  [Truncated to {max_input_tokens} tokens]")
    return report

def build_summarization_prompt(report):
    """Build a clear summarization prompt."""
    return (
        "Please provide a concise and accurate summary of the following government report. "
        "Focus on the key findings, recommendations, and policy implications.\n\n"
        f"REPORT:\n{report}\n\n"
        "SUMMARY:"
    )


# In[ ]:


# Reload model here if you cleared it — or keep model loaded from cell-4
# model, tokenizer = load_model(use_quantization=USE_QUANTIZATION)

# HPC-strong single-pass baseline limits (from dataset profile stats)
MAX_INPUT_TOKENS = 16000  # A100 handles 16k beautifully thanks to FlashAttention
MAX_NEW_TOKENS = 950
TEMPERATURE = 0.3

# Use test split; set None to run full split
test_data = dataset["test"]
NUM_SAMPLES = 100  # Dropped from 120 to 100 to guarantee it fits in 1 hr WITH faithfulness enabled

if NUM_SAMPLES is None:
    run_indices = range(len(test_data))
else:
    run_indices = range(min(NUM_SAMPLES, len(test_data)))

from tqdm.auto import tqdm
results = []

for run_pos, i in enumerate(tqdm(run_indices, desc="Generating Summaries"), start=1):
    sample = test_data[i]
    print(f"\n{'='*60}")
    print(f"Sample {run_pos}/{len(run_indices)} (dataset index {i})")

    # Truncate the source report
    truncated_doc = truncate_report(sample["report"], tokenizer, max_input_tokens=MAX_INPUT_TOKENS)

    # Build prompt and generate
    prompt = build_summarization_prompt(truncated_doc)
    generated_summary = generate_response(
        model, tokenizer,
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE
    )

    results.append({
        "id": i,
        "report": sample["report"],       # original (untruncated) for reference
        "reference_summary": sample["summary"],
        "generated_summary": generated_summary,
    })

    print(f"Reference (first 200 chars): {sample['summary'][:200]}...")
    print(f"Generated (first 200 chars): {generated_summary[:200]}...")

print(f"\nGenerated {len(results)} summaries.")


# In[ ]:


# !pip install rouge-score bert-score

from bert_score import score as bert_score
from rouge_score import rouge_scorer

def compute_rouge(results):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    all_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for r in results:
        scores = scorer.score(r["reference_summary"], r["generated_summary"])
        for key in all_scores:
            all_scores[key].append(scores[key].fmeasure)

    print("=" * 50)
    print("ROUGE SCORES (F1, averaged over samples)")
    print("=" * 50)
    for key, values in all_scores.items():
        avg = sum(values) / len(values)
        print(f"  {key.upper()}: {avg:.4f}")

    return all_scores

rouge_scores = compute_rouge(results)


# ## How the notebook calculates ROUGE
# 
# This cell loops over each item in `results` and compares `reference_summary` against `generated_summary` using `rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)`.
# 
# ### Terms
# - `reference_summary`: the ground-truth summary from the dataset.
# - `generated_summary`: the summary produced by the model.
# - `stemmer=True`: words such as `running` and `run` are reduced to a shared stem before matching.
# - `ROUGE-1`: overlap of 1-grams, meaning single-word tokens.
# - `ROUGE-2`: overlap of 2-grams, meaning consecutive two-word sequences.
# - `ROUGE-L`: overlap based on the Longest Common Subsequence (LCS), which keeps word order but does not require the words to be consecutive.
# - `fmeasure`: the harmonic mean of precision and recall for each ROUGE variant.
# 
# ### Mathematical steps
# For one sample, let:
# - `R` = reference summary tokens
# - `G` = generated summary tokens
# 
# For ROUGE-1, convert both summaries into sets or multisets of 1-grams.
# 
# 1. Count the overlap:
# 
# $$\text{overlap}_1 = |\text{1-grams}(R) \cap \text{1-grams}(G)|$$
# 
# 2. Compute precision:
# 
# $$P_1 = \frac{\text{overlap}_1}{|\text{1-grams}(G)|}$$
# 
# 3. Compute recall:
# 
# $$R_1 = \frac{\text{overlap}_1}{|\text{1-grams}(R)|}$$
# 
# 4. Compute F1:
# 
# $$F1_1 = \frac{2P_1R_1}{P_1 + R_1}$$
# 
# ROUGE-2 uses the exact same formula, but with 2-grams instead of 1-grams:
# 
# $$\text{overlap}_2 = |\text{2-grams}(R) \cap \text{2-grams}(G)|$$
# 
# $$P_2 = \frac{\text{overlap}_2}{|\text{2-grams}(G)|}, \quad R_2 = \frac{\text{overlap}_2}{|\text{2-grams}(R)|}, \quad F1_2 = \frac{2P_2R_2}{P_2 + R_2}$$
# 
# For ROUGE-L, the overlap is based on the LCS length:
# 
# $$L = \text{LCS}(R, G)$$
# 
# $$P_L = \frac{L}{|G|}, \quad R_L = \frac{L}{|R|}, \quad F1_L = \frac{2P_LR_L}{P_L + R_L}$$
# 
# ### What the code stores and reports
# For each sample, the notebook takes `scores[key].fmeasure` for `rouge1`, `rouge2`, and `rougeL`, appends them to `all_scores`, and then averages across all samples:
# 
# $$\text{Average ROUGE} = \frac{1}{N}\sum_{i=1}^{N} F1^{(i)}$$
# 
# So the printed values like `ROUGE1: 0.4051` are mean sample-level F1 scores, not one score computed from concatenating the whole dataset.
# 
# ### Small intuition example
# If the reference is `the cat sat on the mat` and the generated summary is `the cat sat`, then:
# - ROUGE-1 overlap is 3 matching words: `the`, `cat`, `sat`.
# - Precision is `3/3 = 1.0` because every generated word appears in the reference.
# - Recall is `3/6 = 0.5` because only half of the reference words were covered.
# - F1 is `2 * 1.0 * 0.5 / (1.0 + 0.5) = 0.667`.
# This is why ROUGE rewards lexical overlap and coverage, but it does not deeply understand meaning.
# 

# In[ ]:


from bert_score import score as bert_score

def compute_bertscore(results):
    references = [r["reference_summary"] for r in results]
    candidates = [r["generated_summary"] for r in results]

    # Uses DeBERTa by default; rescale_with_baseline improves interpretability
    P, R, F1 = bert_score(
        candidates, references,
        lang="en",
        rescale_with_baseline=True,
        verbose=True
    )

    print("=" * 50)
    print("BERTScore (averaged over samples)")
    print("=" * 50)
    print(f"  Precision: {P.mean().item():.4f}")
    print(f"  Recall:    {R.mean().item():.4f}")
    print(f"  F1:        {F1.mean().item():.4f}")

    return {"precision": P, "recall": R, "f1": F1}

bert_scores = compute_bertscore(results)


# ## How the notebook calculates BERTScore
# 
# This cell builds two lists:
# - `references = [r["reference_summary"] for r in results]`
# - `candidates = [r["generated_summary"] for r in results]`
# 
# It then calls:
# 
# ```python
# P, R, F1 = bert_score(
#     candidates, references,
#     lang="en",
#     rescale_with_baseline=True,
#     verbose=True
# )
# ```
# 
# ### Terms
# - `candidate`: the generated summary we want to evaluate.
# - `reference`: the target summary from the dataset.
# - `embedding`: a dense vector representation of a token produced by a pretrained language model.
# - `cosine similarity`: a number between token vectors measuring semantic similarity.
# - `rescale_with_baseline=True`: shifts and rescales raw similarity scores so they are easier to interpret across examples.
# 
# ### Mathematical steps
# For one sample, suppose the candidate has tokens `c_1, c_2, ..., c_m` and the reference has tokens `r_1, r_2, ..., r_n`.
# 
# 1. Encode each token into contextual embeddings:
# 
# $$\mathbf{e}(c_i), \mathbf{e}(r_j)$$
# 
# 2. Compute pairwise cosine similarity between every candidate token and every reference token:
# 
# $$s_{ij} = \cos(\mathbf{e}(c_i), \mathbf{e}(r_j)) = \frac{\mathbf{e}(c_i) \cdot \mathbf{e}(r_j)}{\|\mathbf{e}(c_i)\|\|\mathbf{e}(r_j)\|}$$
# 
# 3. BERTScore precision asks: for each candidate token, how well does it match the best reference token?
# 
# $$P = \frac{1}{m} \sum_{i=1}^{m} \max_j s_{ij}$$
# 
# 4. BERTScore recall asks: for each reference token, how well does it match the best candidate token?
# 
# $$R = \frac{1}{n} \sum_{j=1}^{n} \max_i s_{ij}$$
# 
# 5. Combine them with the harmonic mean:
# 
# $$F1 = \frac{2PR}{P + R}$$
# 
# ### Why this is different from ROUGE
# ROUGE checks exact token or phrase overlap. BERTScore compares token meanings in embedding space, so synonyms or close paraphrases can still receive a good score even when the exact words differ.
# 
# ### What the code reports
# The function returns one precision, recall, and F1 value per sample. The notebook then averages them:
# 
# $$\bar{P} = \frac{1}{N}\sum_{i=1}^{N} P^{(i)}, \quad \bar{R} = \frac{1}{N}\sum_{i=1}^{N} R^{(i)}, \quad \bar{F1} = \frac{1}{N}\sum_{i=1}^{N} F1^{(i)}$$
# 
# That is what these lines print:
# 
# ```python
# print(f"  Precision: {P.mean().item():.4f}")
# print(f"  Recall:    {R.mean().item():.4f}")
# print(f"  F1:        {F1.mean().item():.4f}")
# ```
# 
# ### Intuition
# If the reference says `the agency reduced emissions` and the candidate says `the agency lowered pollution`, ROUGE may miss part of the match because the words are different, but BERTScore can still give credit if the embedding model treats `reduced` and `lowered`, or `emissions` and `pollution`, as semantically close in context.
# 

# In[ ]:


def self_check_summary(model, tokenizer, report, summary):
    """
    Ask the model to verify if the summary is faithful to the source report.
    Returns a score string: 'Faithful', 'Partially Faithful', or 'Unfaithful'.
    """
    # Keep faithfulness checks cheaper than generation, but not too shallow.
    truncated_doc = truncate_report(report, tokenizer, max_input_tokens=4000)

    verification_prompt = (
        "Given the following source report and its summary, "
        "evaluate whether the summary is factually faithful to the report.\n\n"
        f"report:\n{truncated_doc}\n\n"
        f"SUMMARY:\n{summary}\n\n"
        "Instructions: Does the summary contain only information that is supported by "
        "the report? Answer with one of: 'Faithful', 'Partially Faithful', or 'Unfaithful'. "
        "Then briefly explain why in 1-2 sentences.\n\n"
        "Verdict:"
    )

    verdict = generate_response(
        model, tokenizer,
        verification_prompt,
        max_new_tokens=120,
        temperature=0.1   # Near-deterministic for judgment
    )
    return verdict.strip()


RUN_FAITHFULNESS_CHECK = True # Enabled for the 1-hour one-shot HPC run!
FAITHFULNESS_MAX_SAMPLES = None  # None => evaluate all generated summaries

if RUN_FAITHFULNESS_CHECK:
    print("Running self-checking faithfulness evaluation...")
    eval_results = results if FAITHFULNESS_MAX_SAMPLES is None else results[:FAITHFULNESS_MAX_SAMPLES]

    for i, r in enumerate(eval_results, start=1):
        verdict = self_check_summary(
            model, tokenizer,
            r["report"],
            r["generated_summary"]
        )
        r["faithfulness_verdict"] = verdict
        print(f"\nSample {i}/{len(eval_results)}: {verdict}")
else:
    print("Skipping faithfulness check. Set RUN_FAITHFULNESS_CHECK=True to enable.")


# In[ ]:


import pandas as pd
from rouge_score import rouge_scorer

def full_evaluation_report(results):
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
    print(df.to_string(index=False))
    print("\nAverages:")
    print(df[["rouge1_f1", "rouge2_f1", "rougeL_f1", "gen_length", "ref_length"]].mean())
    return df

df_report = full_evaluation_report(results)


# In[ ]:


# Persist outputs to disk for reproducibility and resume
from datetime import datetime, timezone
from pathlib import Path
import json

RUN_TAG = "baseline_qwen25_7b_a100_100samples_faithful"  # Updated to reflect the 100 sample faithful one-shot run
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_jsonl_path = OUTPUT_DIR / f"{RUN_TAG}_results.jsonl"
results_json_path = OUTPUT_DIR / f"{RUN_TAG}_results.json"
metrics_csv_path = OUTPUT_DIR / f"{RUN_TAG}_per_sample_metrics.csv"
summary_json_path = OUTPUT_DIR / f"{RUN_TAG}_metrics_summary.json"

# Save raw generations
with open(results_jsonl_path, "w", encoding="utf-8") as f:
    for row in results:
        f.write(json.dumps(row) + "\n")

with open(results_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Save per-sample metrics when available
if "df_report" in globals():
    df_report.to_csv(metrics_csv_path, index=False)

summary = {
    "run_tag": RUN_TAG,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "num_samples": len(results),
    "max_input_tokens": MAX_INPUT_TOKENS if "MAX_INPUT_TOKENS" in globals() else None,
    "max_new_tokens": MAX_NEW_TOKENS if "MAX_NEW_TOKENS" in globals() else None,
    "temperature": TEMPERATURE if "TEMPERATURE" in globals() else None,
}

if "df_report" in globals():
    summary.update({
        "rouge1_f1_mean": float(df_report["rouge1_f1"].mean()),
        "rouge2_f1_mean": float(df_report["rouge2_f1"].mean()),
        "rougeL_f1_mean": float(df_report["rougeL_f1"].mean()),
        "gen_length_mean": float(df_report["gen_length"].mean()),
        "ref_length_mean": float(df_report["ref_length"].mean()),
    })

if "bert_scores" in globals():
    summary.update({
        "bertscore_precision_mean": float(bert_scores["precision"].mean().item()),
        "bertscore_recall_mean": float(bert_scores["recall"].mean().item()),
        "bertscore_f1_mean": float(bert_scores["f1"].mean().item()),
    })

with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved artifacts:")
print(f"- {results_jsonl_path}")
print(f"- {results_json_path}")
if "df_report" in globals():
    print(f"- {metrics_csv_path}")
print(f"- {summary_json_path}")
print(f"Rows saved: {len(results)}")


# In[ ]:


"""
# Clear GPU memory
import gc

# Delete model and tokenizer
del model
del tokenizer

# Run garbage collection
gc.collect()

# Clear CUDA cache if using GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU memory cleared")

print("Memory freed successfully!")
"""

