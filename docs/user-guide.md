# User guide

## 1. Choose the prediction unit

A prediction concerns one cell population under one intervention. Measure its control state and provide one or more **support** populations with both control and treated cells for that intervention. The query treated cells are not an input.

Keep biological replicate identities and cell annotations. For evaluation, separate query and support cell types and biological replicate labels, and exclude query treated outcomes from fitting and parameter selection. The evaluated support counts were K = 1–5. More supports can be passed to the API, but their performance has not been established by this release.

## 2. Prepare raw counts

Use nonnegative integer cell-by-gene counts and unique Arabidopsis locus identifiers such as `AT1G01050`. Perform cell-quality filtering, doublet removal and biological annotation before summarization. Pass the full measured transcriptome to the count helper.

For each cell, the helper computes `log1p(10000 * counts / total_counts)` over all supplied genes, then selects and orders model genes. For each population it computes the mean normalized expression, fraction of cells with nonzero counts and sampling variance of each mean. At least two cells are required for estimating variance.

```python
from phytopert import PhytoPert, summarize_counts, build_episode

model = PhytoPert.from_pretrained("checkpoints/arabidopsis")

# Each count matrix has cells in rows and genes in columns.
query = summarize_counts(query_counts, gene_names, model.genes,
                         endpoint_id="replicate_B_query_control")
controls = [summarize_counts(x, gene_names, model.genes,
                            endpoint_id=f"replicate_A_type_{i}_control")
            for i, x in enumerate(control_count_matrices)]
treated = [summarize_counts(x, gene_names, model.genes,
                           endpoint_id=f"replicate_A_type_{i}_treated")
           for i, x in enumerate(treated_count_matrices)]
inputs = build_episode(query, controls, treated)
response, components = model.predict(**inputs, return_components=True)
```

Use the same `endpoint_id` only when reusing the **exact same sampled cells**. Different IDs denote disjoint samples. The helper includes covariance contributions from shared endpoints. If samples overlap partially, supply the corresponding covariance explicitly through the array API. This covariance describes finite-cell sampling.

Unmeasured genes are masked. A gene measured with zero counts remains available. The model requires all 3,597 positions in its fixed order, including masked positions. Keep the complete vocabulary during inference because functional-program pooling uses the full panel. Slice output genes afterward.

## 3. Array input contract

Let B be the batch size, K the number of support populations and G = 3,597. Channel 0 is mean **log1p(CP10k)** expression and channel 1 is detection fraction.

| Array | Shape | Meaning |
|---|---|---|
| `query_control` | B × G × 2 | Query control summary |
| `support_control` | B × K × G × 2 | Support control summaries |
| `support_effect` | B × K × G × 2 | Matched treated summary minus control summary |
| `available` | B × K × G | Boolean intersection of measured query and support genes |
| `support_cov` | B × K × K × G × 2 | Covariance of support response means |

The covariance diagonal contains support response variances. Off-diagonal entries include covariance caused by shared treated or control samples. All arrays must use the checkpoint gene order, listed in `checkpoints/arabidopsis/genes.tsv`. A support may mask genes it lacks. Batch examples must have a common K, so group examples by K when batching.

For fully independent endpoints, the off-diagonal covariance is zero. For supports i and j sharing a control sample, covariance includes that control's variance of the mean. For each support, the diagonal is the sum of treated and control variances. `build_episode` implements these rules from endpoint identities.

## 4. Interpret output

`response[b, g, 0]` is the predicted change in mean log-normalised expression. `response[b, g, 1]` is the predicted change in detection fraction. Add the response to the query control summary to obtain the predicted treated state. A value of zero at an unavailable gene means no supported prediction. Use `components["valid"]` to exclude it from analysis.

The decomposition contains:

- `base`: mean measured support response.
- `gene_correction`: gene-level response correction.
- `program_correction`: correction obtained through functional gene sets.
- `context_correction`: adjustment based on query and support control states.
- `projection_adjustment`: adjustment imposed by feasible treated-state bounds.

Their sum reconstructs the prediction to floating-point precision. Contributions are model explanations, not experimental evidence of gene regulation or causation.

To rank candidate response genes, sort available genes by the absolute predicted expression change. Use `model.primary_positions` when reproducing the 2,050-gene evaluation panel. Such a ranking measures predicted response magnitude, not statistical differential-expression significance.

## 5. Save, batch and run on GPU

```python
import numpy as np
np.savez_compressed("inputs.npz", **inputs)
model = PhytoPert.from_pretrained("checkpoints/arabidopsis", device="cuda:0")
response = model.predict(**inputs)
np.savez_compressed("responses.npz", response=response.cpu().numpy(),
                    genes=np.asarray(model.genes))
```

The checkpoint is loaded entirely from local files. Install an appropriate PyTorch build for your hardware. Small batches are sufficient for this compact model. `predict` disables gradient tracking. The regular `forward` method supports gradients for methodological extensions, with the input contract unchanged.

## Common input problems

| Error | Resolution |
|---|---|
| Duplicate gene identifiers | Resolve duplicate annotations before summarizing counts |
| Empty cells or noninteger counts | Use quality-filtered raw counts |
| Expression outside the supported range | Calculate cell-level log1p(CP10k) before averaging |
| Covariance shape or symmetry error | Check endpoint identities, support order and covariance axes |
| Checksum mismatch | Obtain a fresh copy of the released checkpoint files |
| Unexpected gene responses | Check condition matching, biological annotation, gene order and input scale |

See the [model card](model-card.md) for the applicable biological setting and training-study overlap.

## Run real plant-study examples

Run `python examples/predict_real.py --output real_predictions` from the cloned repository. The script also works from another working directory when called by its absolute path. It uses the bundled receptor-restoration and hypoxia input summaries, writes a gene-level CSV and prediction NPZ for each example, and checks all four outputs against their references. See the [example guide](../examples/data/README.md) for data sources and array contents.

To run one example through the command-line interface:

```bash
phytopert-predict --checkpoint checkpoints/arabidopsis --input examples/data/hypoxia_1_inputs.npz --output prediction_hypoxia.npz
```

Reference predictions demonstrate numerical reproducibility. The four examples are not a replacement for the full manuscript evaluation.
