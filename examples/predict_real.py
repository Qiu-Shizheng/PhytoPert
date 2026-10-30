"""Run and verify predictions for real plant perturbation-study inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from phytopert import PhytoPert


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=root / 'checkpoints/arabidopsis')
    parser.add_argument('--data', type=Path, default=root / 'examples/data')
    parser.add_argument('--output', type=Path, default=Path('real_predictions'))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    manifest = json.loads((args.data / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        actual = hashlib.sha256((args.data / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Example file checksum mismatch: {name}')
    weights_hash = hashlib.sha256((args.checkpoint / 'weights.npz').read_bytes()).hexdigest()
    if weights_hash != manifest['model_sha256']:
        raise ValueError('These reference predictions require the documented Arabidopsis checkpoint')
    model = PhytoPert.from_pretrained(args.checkpoint, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    keys = ('query_control', 'support_control', 'support_effect', 'available', 'support_cov')
    reports = []
    for example in manifest['examples']:
        with np.load(args.data / example['input'], allow_pickle=False) as arrays:
            inputs = {key: arrays[key] for key in keys}
        response, components = model.predict(**inputs, return_components=True)
        prediction = response.cpu().numpy()
        with np.load(args.data / example['reference'], allow_pickle=False) as ref:
            np.testing.assert_array_equal(ref['genes'], np.asarray(model.genes))
            np.testing.assert_allclose(prediction, ref['response'], atol=2e-6, rtol=1e-5)
            difference = float(np.max(np.abs(prediction - ref['response'])))
        np.savez_compressed(args.output / (example['name'] + '_prediction.npz'),
                            response=prediction, genes=np.asarray(model.genes),
                            primary_positions=np.asarray(model.primary_positions),
                            **{k: v.cpu().numpy() for k, v in components.items()})
        valid = components['valid'].cpu().numpy()[0]
        with (args.output / (example['name'] + '_genes.csv')).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['gene', 'expression_response', 'detection_response', 'available'])
            for i, gene in enumerate(model.genes):
                writer.writerow([gene, float(prediction[0, i, 0]), float(prediction[0, i, 1]), bool(valid[i])])
        reports.append(dict(example=example['name'], accession=example['accession'],
                            query_cell_type=example['query_cell_type'],
                            max_reference_difference=difference, status='pass'))
        print(f"{example['name']}: {example['query_cell_type']}; reference difference {difference:.3g}")
    (args.output / 'verification.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(f'Verified {len(reports)} real-study examples. Results: {args.output}')


if __name__ == '__main__':
    main()
