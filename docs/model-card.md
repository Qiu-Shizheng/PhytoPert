# PhytoPert model card

## Intended use

PhytoPert predicts population-level Arabidopsis perturbation responses given a query control population and measured support responses to the same intervention. Its outputs support expression-response analysis, gene prioritisation and inspection of model contributions. The checkpoint is a supervised conditional neural model, with no transcriptome foundation-model pretraining stage.

## Training collection

| Study | Biological setting | Included profiles |
|---|---|---:|
| [GSE198394](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198394) | Root-tip complete MS medium dilution, 0.5× versus 1× | 53,929 |
| [GSE212230](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE212230) | Root brassinosteroid signalling | 115,084 |
| [GSE241573](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE241573) | Root auxin response | 5,587 |
| [GSE268881](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268881) | Root ABA and salt responses | 46,943 |
| [GSE273033](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE273033) | Leaf drought response | 49,680 |
| [GSE290214](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290214) | Rosette drought, training augmentation | 133,250 |

The included collection contains 404,473 profiles and 317 eligible response units. These profile counts describe the included subsets, not total cells in the original deposits. MS dilution changes the complete growth medium and is not a specific nutrient-starvation experiment.

The released checkpoint is the model fitted on the combined collection for application use. Published study-exclusion comparisons use separately fitted checkpoints. Evaluating the released combined-data checkpoint on its training studies does not reproduce held-study evaluation.

## Evaluated settings

Conventional tasks include response prediction, cell-type response contrasts, intervention contrasts and limited support-cell sampling. Plant applications include epidermal BRI1 restoration in GSE212230 and root hypoxia in [GSE308672](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE308672). The BRI1 restoration application shares a study with the training collection, using a distinct contrast. Hypoxia uses root nuclei exposed to 4% oxygen for four hours.

Sampling comparisons average four studies and K = 1–5 equally. The three regimes use all available support cells, 48 cells per support population per condition, or 48 cells per condition distributed across support populations. The last regime normally comprises 96 control-plus-treated cells in total. Plant application summaries use K = 3.

| Application | Response MSE ↓ | Response Pearson r ↑ | Top-100 response-gene recovery ↑ |
|---|---:|---:|---:|
| BRI1 restoration | 0.05600 | 0.313 | 28.9% |
| Hypoxia | 0.01685 | 0.542 | 37.5% |

Gene recovery is overlap with the 100 largest observed absolute expression changes. It is not a significance-based differential-expression metric. The model achieved the lowest response MSE among the evaluated comparisons in hypoxia. In BRI1 restoration, it was competitive with perturbation models, including configurations with lower error.

Comparisons include scGen, scVIDR, scPRAM, State, CellFlow, scPlantLLM and UCE, alongside statistical and machine-learning predictors. scPlantLLM and UCE use frozen representations with fixed or fitted kernel prediction rules. State was trained on the project data. Input panels differ: PhytoPert uses 3,597 genes and comparison models use 2,050, with evaluation on the common 2,050-gene panel. These are evaluations of the specified interfaces and training settings.

## Generalisation and interpretation

Individual study-exclusion fits omit the corresponding held study. Two designated studies, GSE212230 and GSE268881, were used to calibrate the mixture weight. The evaluation collection informed method development. Results therefore describe this evaluated collection, and do not establish prospective independent-study validation.

The released model requires measured responses to the same intervention. It predicts population means and detection fractions. Cross-species transfer, response prediction with no measured support, time-resolved dynamics and full single-cell distributions are outside its evaluated scope. Functional gene sets also participate in the architecture, so their response agreement is not independent validation of a biological mechanism. Model contributions can guide follow-up experiments but do not establish causal regulatory interactions.

## Technical release

Weights are stored as numeric NPZ arrays, with JSON configuration and SHA-256 checksums. Loading uses `allow_pickle=False`. The release includes the full fixed gene vocabulary, primary evaluation positions and trained functional memberships. The [validation record](validation.json) documents equivalence to saved predictions and tested software versions.
