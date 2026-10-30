<div align="center">

# PhytoPert

**Predicting plant perturbation responses across cell populations**

[User guide](docs/user-guide.md) · [Model card](docs/model-card.md) · [Architecture](docs/architecture.md) · [Weights](checkpoints/arabidopsis)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Framework](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C)
![Species](https://img.shields.io/badge/Species-Arabidopsis%20thaliana-007F5F)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

PhytoPert predicts how an unmeasured plant cell population responds to an intervention, using its control state and responses measured in other cell populations exposed to the **same intervention**. It combines gene, functional-program and cell-context modules in a compact neural model.

For each gene, PhytoPert predicts the change in **mean log-normalised expression** and **expressing-cell fraction**. Its additive components help users examine how measured responses and learned corrections contribute to a prediction.

<p align="center"><img src="docs/assets/architecture.svg" alt="PhytoPert model architecture" width="900"></p>

### What you can do

- Extend a partially measured perturbation experiment to additional cell populations.
- Predict response magnitude and direction with limited support-cell sampling.
- Rank genes for targeted expression measurements.
- Inspect gene, functional-program and cell-context contributions.

The released checkpoint is trained for **Arabidopsis thaliana**, uses **3,597 genes** and contains **100,820 parameters** across two complementary operators. The primary evaluation panel contains 2,050 genes. CPU inference is supported.

### Quick start

```bash
git clone https://github.com/Qiu-Shizheng/PhytoPert.git
cd PhytoPert
python -m pip install -e .
python examples/predict_real.py --output real_predictions
```

This runs four real-data examples from the epidermal BRI1 restoration and root hypoxia studies. It saves per-gene response tables, verifies predictions against the supplied references and records the checks in `real_predictions/verification.json`. Model weights and example inputs are included. Inference does not download anything.

Read the [real-data example guide](examples/data/README.md) for study provenance and input details. `python examples/predict.py` provides a separate synthetic-count example of preprocessing your own data.

```python
from phytopert import PhytoPert

model = PhytoPert.from_pretrained("checkpoints/arabidopsis")
response, components = model.predict(
    query_control=query_control,       # [batch, genes, 2]
    support_control=support_control,   # [batch, supports, genes, 2]
    support_effect=support_effect,     # treated minus control
    available=available,               # boolean [batch, supports, genes]
    support_cov=support_cov,           # [batch, supports, supports, genes, 2]
    return_components=True,
)
```

Use `summarize_counts` and `build_episode` to prepare these arrays from raw counts. The [user guide](docs/user-guide.md) explains normalization, gene matching, biological replicates and shared controls.

### Command-line prediction

```bash
phytopert-predict \
  --checkpoint checkpoints/arabidopsis \
  --input examples/data/hypoxia_1_inputs.npz \
  --output prediction_cli.npz
```

The output includes ordered gene identifiers, two-channel responses, an availability mask and additive components. Use `--device cuda:0` with a CUDA-enabled PyTorch installation for GPU inference.

### Architecture

Two response operators use complementary gene representations: learned gene embeddings and biological descriptors. Each combines the measured mean support response with gene, functional-program and cell-context corrections. Functional modules pool and redistribute information through 12 fixed Gene Ontology gene sets. Both operators constrain the predicted treated state to feasible ranges before equal averaging.

The model uses control states and measured support responses. Its supported task is conditional prediction of population-level perturbation responses. See the [model card](docs/model-card.md) for training data, evaluation scope and interpretation.

### Repository contents

| Location | Contents |
|---|---|
| `phytopert/` | Portable model, count summarization and command-line interface |
| `checkpoints/arabidopsis/` | Numeric weights, gene vocabulary, configuration and checksums |
| `examples/predict_real.py` | Four verified real-study predictions and per-gene CSV outputs |
| `examples/data/` | Aggregate study inputs, reference predictions and provenance |
| `examples/predict.py` | Synthetic counts illustrating preprocessing |
| `tests/` | Input, covariance, output and checkpoint-integrity tests |
| `docs/` | User guide, architecture, model card and export validation |

### Verify your installation

```bash
python -m unittest discover -s tests -v
```

Export validation compares the released model against saved study predictions. Details are recorded in [validation.json](docs/validation.json).

### Citation and licence

Please cite this software repository and identify the release used in your analysis. A software citation is provided in [CITATION.cff](CITATION.cff). Code and model weights are distributed under the [MIT licence](LICENSE). Original expression datasets remain available from their study repositories and retain their respective terms.

For questions about the interface or reproducible examples, open a [GitHub issue](https://github.com/Qiu-Shizheng/PhytoPert/issues).
