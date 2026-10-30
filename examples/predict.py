"""Run an explicitly synthetic example; these data have no biological meaning."""
import argparse
from pathlib import Path
import numpy as np
from phytopert import PhytoPert, summarize_counts, build_episode

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="checkpoints/arabidopsis")
parser.add_argument("--output", default="prediction_example.npz")
args = parser.parse_args()
model = PhytoPert.from_pretrained(args.checkpoint)
rng = np.random.default_rng(42)
genes = list(model.genes) + ["EXAMPLE_EXTRA_GENE"]
rate = rng.uniform(.1, 2., len(genes))
def population(name, treated=False):
    values = rate.copy()
    if treated:
        values[:50] *= 1.3
    return summarize_counts(rng.poisson(values, size=(48, len(genes))),
                            genes, model.genes, endpoint_id=name)
query = population("query_control")
controls = [population(f"support_{i}_control") for i in range(3)]
treated = [population(f"support_{i}_treated", True) for i in range(3)]
episode = build_episode(query, controls, treated)
response, components = model.predict(**episode, return_components=True)
np.savez_compressed(args.output, response=response.cpu().numpy(), genes=np.asarray(model.genes),
                    **{name: x.cpu().numpy() for name, x in components.items()})
np.savez_compressed(Path(args.output).with_name("example_inputs.npz"), **episode)
print(f"Synthetic demonstration: {tuple(response.shape)}, saved to {args.output}")
