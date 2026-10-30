"""Portable PhytoPert inference and additive response explanations."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ._operator import ResponseOperator


class DescriptorEmbedding(nn.Module):
    def __init__(self, descriptors):
        super().__init__()
        self.register_buffer("descriptors", descriptors)
        self.encoder = nn.Linear(descriptors.shape[1], 16, bias=False)

    def forward(self, ids):
        return .02 * self.encoder(self.descriptors[ids])


class PhytoPert(nn.Module):
    """Predict population responses from controls and measured support effects.

    Channel 0 is mean log1p(CP10k). Channel 1 is the expressing-cell fraction.
    Load the released Arabidopsis model with :meth:`from_pretrained`.
    """

    def __init__(self, config, state):
        super().__init__()
        self.config = config
        self.genes = tuple(config["genes"])
        self.primary_positions = tuple(config["primary_positions"])
        self.alpha = float(config["mixture_weight"])
        for name in ("learned", "descriptor"):
            assets = {
                "fit_gene_ids": state[f"{name}.parent.seen_genes"].nonzero().flatten(),
                "program_membership": state[f"{name}.program_membership"],
                "response_scale": state[f"{name}.response_units"][:, 0],
            }
            branch = ResponseOperator(assets=assets, genes=len(self.genes))
            if name == "descriptor":
                branch.parent.embedding = DescriptorEmbedding(
                    state["descriptor.parent.embedding.descriptors"].clone())
            setattr(self, name, branch)
        self.load_state_dict(state, strict=True)

    @classmethod
    def from_pretrained(cls, directory, device="cpu"):
        """Load local, checksum-verified JSON and NPZ files. No network access."""
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        for name in ("config.json", "weights.npz"):
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
            if actual != manifest["sha256"][name]:
                raise ValueError(f"Checksum mismatch: {name}")
        config = json.loads((root / "config.json").read_text())
        if config.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        with np.load(root / "weights.npz", allow_pickle=False) as archive:
            state = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
        model = cls(config, state).to(device).eval()
        if sum(p.numel() for p in model.parameters()) != config["parameters"]:
            raise ValueError("Checkpoint parameter count does not match configuration")
        return model

    def _inputs(self, query_control, support_control, support_effect, available, support_cov):
        device = next(self.parameters()).device
        def tensor(value, dtype=torch.float32):
            return torch.as_tensor(value, dtype=dtype, device=device)
        qc, sc, sy, cov = map(tensor, (query_control, support_control, support_effect, support_cov))
        av = torch.as_tensor(available, device=device)
        if av.dtype != torch.bool:
            raise ValueError("available must be a boolean gene mask")
        g = len(self.genes)
        if qc.ndim != 3 or qc.shape[1:] != (g, 2):
            raise ValueError(f"query_control must have shape (batch, {g}, 2)")
        if sc.ndim != 4 or sc.shape != sy.shape or sc.shape[0] != qc.shape[0] or sc.shape[2:] != (g, 2):
            raise ValueError("support_control and support_effect must have shape (batch, supports, genes, 2)")
        b, k = sc.shape[:2]
        if k < 1 or av.shape != (b, k, g) or cov.shape != (b, k, k, g, 2):
            raise ValueError("Inconsistent support count, mask or covariance shape")
        if not av.any(-1).all():
            raise ValueError("Each support must have measured genes shared with the query")
        mask = av[:, :, None, :, None] & av[:, None, :, :, None]
        measured_cov = torch.where(mask, cov, torch.zeros_like(cov))
        if not torch.isfinite(measured_cov).all() or not torch.allclose(measured_cov, measured_cov.transpose(1, 2), atol=1e-7, rtol=1e-5):
            raise ValueError("Measured support covariance must be finite and symmetric")
        diagonal = measured_cov.diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
        if (diagonal < 0).any():
            raise ValueError("Sampling variances must be nonnegative")
        bound = (diagonal[:, :, None] * diagonal[:, None]).sqrt()
        if (measured_cov.abs() > bound + 1e-6).any():
            raise ValueError("Sampling covariance exceeds the variance bound")
        for values, valid in ((qc, av.any(1)), (sc, av), (sc + sy, av)):
            selected = values[valid]
            if not torch.isfinite(selected).all():
                raise ValueError("Measured summaries must be finite")
            if (selected[:, 0] < -1e-5).any() or (selected[:, 0] > np.log1p(10000) + 1e-5).any():
                raise ValueError("Expression summaries must use mean log1p(CP10k)")
            if (selected[:, 1] < -1e-5).any() or (selected[:, 1] > 1 + 1e-5).any():
                raise ValueError("Detection fractions must be between zero and one")
        return qc, sc, sy, av, cov

    def forward(self, query_control, support_control, support_effect, available, support_cov,
                return_components=False):
        """Return a (batch, 3597, 2) response tensor, optionally with components.

        The covariance shape is (batch, supports, supports, 3597, 2).
        It describes finite-cell sampling of support effects, including shared
        endpoints. Output genes with no available support have zero response.
        """
        qc, sc, sy, av, cov = self._inputs(query_control, support_control, support_effect, available, support_cov)
        ids = self.learned.full_gene_ids
        p, a = self.learned(ids, qc, sc, sy, av, support_cov=cov, return_explanation=return_components)
        q, b = self.descriptor(ids, qc, sc, sy, av, support_cov=cov, return_explanation=return_components)
        prediction = (1 - self.alpha) * p + self.alpha * q
        if not return_components:
            return prediction
        keys = ("base", "gene_correction", "program_correction", "context_correction",
                "unclamped_prediction", "projection_adjustment")
        components = {key: (1-self.alpha)*a[key] + self.alpha*b[key] for key in keys}
        components["valid"] = av.any(1)
        return prediction, components

    @torch.inference_mode()
    def predict(self, **inputs):
        """Run inference without gradient tracking. Accept NumPy arrays or tensors."""
        return self(**inputs)
