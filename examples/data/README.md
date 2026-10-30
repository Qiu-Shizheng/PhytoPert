# Real plant perturbation examples

These files contain aggregate inputs from two public Arabidopsis studies used in the manuscript. Each input holds one query control state and three measured support responses to the same intervention. All arrays follow the 3,597-gene order in the released checkpoint.

| Files | Study | Perturbation |
|---|---|---|
| `rescue_1_*`, `rescue_2_*` | [GSE212230](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE212230) | Epidermal BRI1 receptor restoration using pGL2-BRI1-GFP in the bri1-T background |
| `hypoxia_1_*`, `hypoxia_2_*` | [GSE308672](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE308672) | Root nuclei after 4 h at 4% oxygen, compared with 21% oxygen |

`manifest.json` identifies the query and support cell types, sample labels, population sizes, array filenames and checksums. The examples are the first two distinct query contexts in each study's fixed K = 3 evaluation plan, using all eligible support cells. They were selected by plan order.

## Contents

- `*_inputs.npz`: query control, support controls, support effects, gene-availability mask and sampling covariance. Query treated outcomes are excluded.
- `*_reference.npz`: model predictions, ordered genes, evaluation-panel positions and additive components.
- `manifest.json`: biological metadata and file-integrity checks.

The reference arrays are **model predictions**, not observed treated expression. They verify numerical reproducibility of inference. Full application-level accuracy is evaluated across all query contexts and support draws in the manuscript.

Run all four examples from the repository:

```bash
python examples/predict_real.py --output real_predictions
```

No network access is needed for inference after installation. The script verifies file checksums, compares outputs with reference predictions, and writes per-gene CSV files and NPZ predictions. Positive expression responses indicate increased mean log1p CP10k expression; detection responses describe changes in the expressing-cell fraction.

## Attribution

The input summaries derive from the original public expression datasets linked above. Please cite the corresponding original studies through their GEO records when reusing the data. These summaries do not replace the full study deposits or their annotations. The repository's MIT licence covers its software and model weights; original study data retain their respective terms.
