"""Predict from a portable NPZ input file."""
import argparse
import numpy as np
from .model import PhytoPert


def main():
    parser = argparse.ArgumentParser(description="Predict plant population responses with PhytoPert")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="NPZ with the five documented input arrays")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = PhytoPert.from_pretrained(args.checkpoint, device=args.device)
    keys = ("query_control", "support_control", "support_effect", "available", "support_cov")
    with np.load(args.input, allow_pickle=False) as data:
        missing = set(keys) - set(data.files)
        if missing:
            parser.error("Missing input arrays: " + ", ".join(sorted(missing)))
        inputs = {key: data[key] for key in keys}
    pred, components = model.predict(**inputs, return_components=True)
    np.savez_compressed(args.output, response=pred.cpu().numpy(),
                        genes=np.asarray(model.genes), primary_positions=np.asarray(model.primary_positions),
                        **{key: value.cpu().numpy() for key, value in components.items()})
    print(f"Saved {pred.shape[0]} prediction(s) for {len(model.genes)} genes")


if __name__ == "__main__":
    main()
