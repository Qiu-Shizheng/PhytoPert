"""Raw-count summaries and shared-endpoint sampling covariance."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class CellSummary:
    mean: np.ndarray
    variance_of_mean: np.ndarray
    measured: np.ndarray
    n_cells: int
    endpoint_id: str


def summarize_counts(counts, gene_names, model_genes, endpoint_id):
    """Summarize raw cell-by-gene counts for one biological endpoint.

    Normalize each cell over ALL provided genes before selecting model genes.
    Supply the full measured transcriptome and one cell population, condition
    and biological replicate per endpoint. Reuse the endpoint ID only for the
    exact same sampled cells. At least two cells are needed for sample variance.
    """
    names = list(gene_names)
    if not endpoint_id or not isinstance(endpoint_id, str):
        raise ValueError("A nonempty endpoint_id is required")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate gene identifiers: resolve them before summarization")
    x = sparse.csr_matrix(counts, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(names) or x.shape[0] < 2:
        raise ValueError("Expected at least two cells and one column per gene identifier")
    x.sum_duplicates()
    if not np.isfinite(x.data).all() or (x.data < 0).any() or not np.equal(x.data, np.floor(x.data)).all():
        raise ValueError("Input must contain finite, nonnegative raw integer counts")
    x.eliminate_zeros()
    total = np.asarray(x.sum(1)).ravel()
    if (total <= 0).any():
        raise ValueError("Remove empty cells before summarization")
    logx = x.multiply((10000 / total)[:, None]).tocsr()
    logx.data = np.log1p(logx.data)
    mu = np.asarray(logx.mean(0)).ravel()
    second = np.asarray(logx.power(2).mean(0)).ravel()
    detect = np.asarray((x > 0).mean(0)).ravel()
    # unbiased sample variance / N = (E[X^2] - E[X]^2) / (N - 1)
    variance = np.stack((np.maximum(0, second - mu**2), detect * (1-detect)), -1) / (x.shape[0]-1)
    idx = {gene: i for i, gene in enumerate(names)}
    measured = np.array([gene in idx for gene in model_genes], dtype=bool)
    positions = [idx[gene] for gene in model_genes if gene in idx]
    mean = np.zeros((len(model_genes), 2), dtype=np.float32)
    var = np.zeros_like(mean)
    mean[measured] = np.stack((mu, detect), -1)[positions]
    var[measured] = variance[positions]
    return CellSummary(mean, var, measured, x.shape[0], endpoint_id)


def build_episode(query_control, support_controls, support_treated):
    """Build a batch of one, accounting for exactly reused support endpoints.

    Different endpoint IDs denote disjoint sampled cells. Partially overlapping
    samples require a user-supplied covariance tensor via the low-level API.
    All support pairs must represent the same intervention and matched contexts.
    """
    controls, treated = list(support_controls), list(support_treated)
    if not controls or len(controls) != len(treated):
        raise ValueError("Provide one control and one treated endpoint per support")
    seen = {}
    for endpoint in controls + treated:
        if endpoint.mean.shape != query_control.mean.shape:
            raise ValueError("Summaries must share the complete ordered model vocabulary")
        previous = seen.get(endpoint.endpoint_id)
        if previous is not None:
            if previous.n_cells != endpoint.n_cells or any(not np.array_equal(getattr(previous, k), getattr(endpoint, k)) for k in ("mean", "variance_of_mean", "measured")):
                raise ValueError("The same endpoint_id refers to different sampled cells")
        seen[endpoint.endpoint_id] = endpoint
    if query_control.endpoint_id in seen:
        raise ValueError("Query controls must be separate from support endpoints")
    terms = []
    for c, t in zip(controls, treated):
        if c.endpoint_id == t.endpoint_id:
            raise ValueError("Control and treated endpoints must be distinct")
        terms.append({c.endpoint_id: -1, t.endpoint_id: 1})
    k, g = len(controls), len(query_control.mean)
    covariance = np.zeros((1, k, k, g, 2), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            for endpoint in terms[i].keys() & terms[j].keys():
                covariance[0, i, j] += terms[i][endpoint] * terms[j][endpoint] * seen[endpoint].variance_of_mean
    return dict(query_control=query_control.mean[None],
                support_control=np.stack([c.mean for c in controls])[None],
                support_effect=np.stack([t.mean-c.mean for c,t in zip(controls,treated)])[None],
                available=np.stack([query_control.measured & c.measured & t.measured for c,t in zip(controls,treated)])[None],
                support_cov=covariance)
