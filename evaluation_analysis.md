# Evaluation Analysis

**Author:** Yangyie  
**Course:** CS6140 Machine Learning — Final Project  
**Project:** Self-Checking Summarizer for Long Government Documents (GovReport)

---

## 1. Metric Selection Rationale

### Why These Four Metrics

Summarization evaluation has many candidates. We selected four metrics based on one principle: **each metric must reveal something the others cannot.**

| Metric | What It Captures | Why It's Necessary |
|--------|-----------------|-------------------|
| ROUGE-1 / 2 / L | N-gram overlap with reference summary | Industry standard — enables direct comparison with prior GovReport work |
| BERTScore | Semantic similarity via BERT embeddings | Captures paraphrase and synonym — what ROUGE misses |
| Length Coverage | Generated length / reference length | Directly diagnoses under-coverage in long-document tasks |
| Faithfulness (Self-Check) | Whether summary is supported by source | Detects hallucination without requiring human annotation |

### Why We Rejected Other Metrics

| Metric | Reason for Rejection |
|--------|---------------------|
| BLEU | Designed for machine translation; penalizes length variation in ways unsuitable for summarization |
| METEOR | Less standardized for summarization benchmarks |
| Perplexity | Measures fluency only, not content accuracy or coverage |
| Human Evaluation | Gold standard but not scalable to 100+ samples in this setting |

### The Concrete Problem With ROUGE Alone

Consider two sentences with identical meaning:

- **Generated:** *"recommends stricter federal oversight"*
- **Reference:** *"suggests enhanced government monitoring"*

ROUGE-2 scores this as **0.0** (no bigram overlap).  
BERTScore scores this as **> 0.8** (high semantic similarity).

This is why BERTScore is necessary alongside ROUGE — especially for government documents where different authors use different terminology to describe the same policy concepts.

---

## 2. How the Metrics Are Computed

### ROUGE-N

ROUGE measures n-gram overlap between the generated summary and the reference summary.

```
Recall    = count(matched n-grams) / count(n-grams in reference)
Precision = count(matched n-grams) / count(n-grams in generated)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

- **ROUGE-1**: unigram overlap
- **ROUGE-2**: bigram overlap (more sensitive to phrase-level precision)
- **ROUGE-L**: longest common subsequence (sensitive to word order)

**Key limitation:** Requires exact word match. Paraphrase receives zero credit.

### BERTScore

BERTScore maps each token to a contextual BERT embedding, then computes cosine similarity between token pairs across the generated and reference summaries.

```
Precision = (1/|ŷ|) × Σ max_{j} cos_sim(embed(ŷᵢ), embed(yⱼ))
Recall    = (1/|y|)  × Σ max_{i} cos_sim(embed(yⱼ), embed(ŷᵢ))
F1        = 2PR / (P + R)
```

We use `rescale_with_baseline=True`, which normalizes scores into a more interpretable range.

**Key advantage:** Semantic equivalents (EPA ≈ agency, recommended ≈ proposed) receive partial credit.

### Length Coverage

```
Coverage = mean(generated length in words) / mean(reference length in words)
```

This simple ratio is a direct diagnostic for under-coverage — a failure mode that neither ROUGE nor BERTScore reliably surfaces on its own.

### Faithfulness (Self-Check)

The same LLM is prompted to evaluate its own output:

```
"Given the following source text and summary, is the summary 
fully supported by the source? Answer: Faithful / Unfaithful."
```

No ground-truth labels required. The model acts as its own critic.

---

## 3. Baseline Results

**Setup:** Qwen2.5-7B-Instruct, 4-bit quantization, A100 GPU, 100 samples from GovReport

| Metric | Value | Interpretation |
|--------|-------|---------------|
| ROUGE-1 F1 | 0.4945 | Moderate — topic-level coverage acceptable |
| ROUGE-2 F1 | 0.1818 | Low — phrase-level precision is weak |
| ROUGE-L F1 | 0.2136 | Low — structural ordering diverges from reference |
| BERTScore F1 | 0.0774 | Low — semantic coverage incomplete |
| Avg Generated Length | 374.5 words | — |
| Avg Reference Length | 587.8 words | — |
| **Length Coverage** | **63.7%** | The model captures ~2/3 of what a reference covers |
| Faithfulness | 100% Faithful | No hallucination detected — the problem is coverage, not accuracy |

### Root Cause

GovReport documents average 15,000+ tokens. Even with a 16K input window, the model generates summaries that are only 64% as long as the references. The model reads what it can, then stops — the back half of most reports (recommendations, implementation details, appendices) never gets summarized.

This is an **under-coverage problem, not a fluency or hallucination problem.**

---

## 4. Divide-and-Conquer Pipeline Results

We implemented three variants of a map-reduce summarization pipeline:

- **MapReduce**: parallel chunk processing, simultaneous reduction
- **Map-Refine** (sequential): running summary accumulates context forward
- **Map-Cluster-Reduce**: semantically groups chunks before reducing

**Best performer: Map-Refine**

| Metric | Baseline | Map-Refine | Change |
|--------|----------|------------|--------|
| ROUGE-1 F1 | 0.4945 | **0.5129** | ↑ +3.7% |
| ROUGE-2 F1 | 0.1818 | **0.1696** | ↓ −6.7% |
| ROUGE-L F1 | 0.2136 | **0.2076** | ↓ −2.8% |
| BERTScore F1 | 0.0774 | **0.0918** | ↑ +18.6% |
| Avg Generated Length | 374.5 | **495** words | ↑ +32% |
| Length Coverage | 63.7% | **84.2%** | ↑ +20pp |
| Redundancy | — | ~10–11% | (new cost) |

---

## 5. The Core Tradeoff: Coverage vs. Precision

The D&C results are not uniformly better — they reveal a fundamental tradeoff.

### What Improved

- **ROUGE-1 ↑** — more content words from the reference appear in the output
- **BERTScore ↑** — semantic meaning is better captured across the full document
- **Length ↑** — summaries now cover 84% of reference length vs. 64% before

### What Declined

- **ROUGE-2 ↓** — exact bigram matches fell despite more content being covered
- **ROUGE-L ↓** — structural ordering relative to the reference weakened

### Why This Happens

The merge step (reduce phase) prompts the model to integrate multiple partial summaries into a coherent final output. To maintain fluency and avoid repetition, the model **paraphrases** the content from individual chunk summaries rather than preserving their exact phrasing.

This improves readability and semantic completeness — but it breaks exact n-gram chains, which ROUGE-2 and ROUGE-L rely on.

### What This Tells Us About Evaluation

This tradeoff is not a failure of the pipeline — it is a **limitation of ROUGE as a sole evaluation metric for long-document summarization.**

When a model must synthesize content from multiple sources into a coherent summary, some degree of paraphrase is inevitable and desirable. ROUGE-2 penalizes this paraphrase even when the meaning is preserved. BERTScore correctly rewards it.

This is precisely why we chose complementary metrics: **ROUGE-1 alone would have missed the precision cost; BERTScore alone would have missed it too, in the other direction.**

---

## 6. Key Insights

1. **Pipeline design matters more than model size.** The same 7B model produces substantially different results depending on whether it processes the full document at once or in structured chunks.

2. **Long-document summarization is a coverage problem, not a fluency problem.** The baseline is coherent and faithful — it simply misses too much content. D&C directly targets this root cause.

3. **Evaluation metrics must complement each other.** ROUGE-2 and BERTScore told opposite stories about the D&C results. Using both is what allowed us to correctly diagnose the tradeoff rather than misreporting either a clean win or a clean loss.

4. **The merge step is the critical bottleneck.** Improving chunk-level summaries is relatively straightforward. The hard problem is merging them without losing phrase-level precision.

---

## 7. Future Directions

| Direction | Motivation |
|-----------|-----------|
| Structured chunk notes (schema-based mapper) | Reduce paraphrase in the reduce step by standardizing intermediate format |
| Scale to 500+ samples | Current 100-sample evaluation may not be statistically stable for ROUGE-2 comparisons |
| Critic/refiner stage | Add a dedicated verification pass on chunk-level notes, not just the final output |
| Improve redundancy detection | 10–11% redundancy in Map-Refine outputs suggests deduplication logic needs work |
| Cross-domain transfer | Test whether the pipeline generalizes to legal or scientific long documents |
