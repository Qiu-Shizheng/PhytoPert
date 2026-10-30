"""Gene, functional-program and cell-context response modules."""
import math
import torch
from torch import nn
ARMS = ('program_context',)

def project_delta(delta, query_control):
    upper = torch.stack((torch.full_like(query_control[..., 0], math.log1p(10000)), torch.ones_like(query_control[..., 1])), dim=-1) - query_control
    return torch.maximum(-query_control, torch.minimum(delta, upper))

def _mlp(inputs, width, outputs, zero=False):
    net = nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, outputs, bias=False))
    if zero:
        nn.init.zeros_(net[-1].weight)
    return net

def _anchored(net, condition, effect):
    return net(torch.cat((condition, effect), -1)) - net(torch.cat((condition, torch.zeros_like(effect)), -1))

def _signed_log(x):
    return x.sign() * x.abs().log1p()

def _mean_genes(x, valid):
    return (x * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)

class _ProgramBackbone(nn.Module):

    def __init__(self, genes, programs, fit_gene_ids):
        super().__init__()
        self.gene_encoder = _mlp(16 + 7 + 4, 64, 16)
        self.gene_decoder = _mlp(16 + 7 + 32, 64, 2, zero=True)
        self.program_encoder = _mlp(programs + 16, 32, 16)
        self.program_decoder = _mlp(16 + programs + 7 + 32, 64, 2, zero=True)
        self.common_gain = nn.Parameter(torch.zeros(2))
        self.embedding = nn.Embedding(genes + 1, 16, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        seen = torch.zeros(genes, dtype=torch.bool)
        fit = torch.as_tensor(fit_gene_ids, dtype=torch.long).cpu().reshape(-1)
        if ((fit < 0) | (fit >= genes)).any():
            raise ValueError('fit_gene_ids outside the fixed gene vocabulary')
        seen[fit] = True
        self.register_buffer('seen_genes', seen)

    def gene_features(self, ids):
        mapped = torch.where(self.seen_genes[ids], ids + 1, torch.zeros_like(ids))
        return self.embedding(mapped)

class ResponseOperator(nn.Module):

    def __init__(self, arm='program_context', fit_gene_ids=None, assets=None, genes=3597):
        if arm not in ARMS:
            raise ValueError(arm)
        if fit_gene_ids is None:
            if assets is None or 'fit_gene_ids' not in assets:
                raise ValueError('Training gene identities must be supplied')
            fit_gene_ids = assets['fit_gene_ids']
        nn.Module.__init__(self)
        self.arm, self.genes = (arm, int(genes))
        self.baseline_kind = 'unit_gain_support_mean'
        if assets is None or 'program_membership' not in assets or 'response_scale' not in assets:
            raise ValueError('Program arms need fixed program_membership and training response_scale')
        membership = torch.as_tensor(assets['program_membership'], dtype=torch.float32).detach().cpu().clone()
        if membership.ndim != 2 or membership.shape[0] != genes or membership.shape[1] < 1:
            raise ValueError('program_membership must have shape genes, programs')
        if not torch.isfinite(membership).all() or not ((membership == 0) | (membership == 1)).all():
            raise ValueError('Program membership must be finite and binary')
        if (membership.sum(0) == 0).any():
            raise ValueError('Each frozen program must contain at least one panel gene')
        permutation = torch.arange(genes)
        scale = torch.as_tensor(assets['response_scale'], dtype=torch.float32).detach().cpu().clone()
        if scale.shape != (genes,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('response_scale must be positive, finite, and training')
        self.register_buffer('program_membership', membership)
        self.register_buffer('program_permutation', permutation)
        self.register_buffer('response_units', torch.stack((scale, torch.full_like(scale, 0.2)), -1))
        self.register_buffer('full_gene_ids', torch.arange(genes))
        self.register_buffer('control_units', torch.tensor([3.0, 1.0]))
        programs = membership.shape[1]
        self.register_buffer('program_identity', torch.eye(programs))
        self.parent = _ProgramBackbone(genes, programs, fit_gene_ids)
        if arm != 'program_shared':
            self.context_encoder = _mlp(18, 32, 16)
            self.context_decoder = _mlp(73 + 48, 64, 2, zero=True)

    def parameter_counts(self):
        total = sum((p.numel() for p in self.parameters()))
        context = sum((p.numel() for n, p in self.named_parameters() if n.startswith('context_')))
        program = sum((p.numel() for n, p in self.named_parameters() if n.startswith('parent.program_')))
        return dict(allocated=total, trainable=sum((p.numel() for p in self.parameters() if p.requires_grad)), backbone=sum((p.numel() for p in self.parent.parameters())), program_modules=program, context_adapter=context, local_adapter=0, genes=self.genes, baseline_kind=self.baseline_kind)

    def _clean(self, ids, qc, sc, sy, available):
        if qc.ndim != 3 or qc.shape[-1] != 2 or sc.ndim != 4 or (sc.shape != sy.shape):
            raise ValueError('qc must be B,G,2 and sc/sy B,K,G,2')
        if sc.shape[-1] != 2 or sc.shape[1] < 1 or sc.shape[0] != qc.shape[0] or (sc.shape[2] != qc.shape[1]):
            raise ValueError('Inconsistent query/support dimensions')
        if available.dtype != torch.bool or available.shape != sc.shape[:-1]:
            raise ValueError('available must be bool B,K,G')
        if len(ids) != self.genes or qc.shape[1] != self.genes or (not torch.equal(ids, self.full_gene_ids)):
            raise ValueError('Program pooling requires the complete ordered vocabulary; slice outputs only')
        valid = available.any(1)
        qc = torch.where(valid[..., None], qc.float(), torch.zeros_like(qc).float())
        sc = torch.where(available[..., None], sc.float(), torch.zeros_like(sc).float())
        sy = torch.where(available[..., None], sy.float(), torch.zeros_like(sy).float())
        if not torch.isfinite(qc).all() or not torch.isfinite(sc).all() or (not torch.isfinite(sy).all()):
            raise ValueError('Nonfinite control or response at a measured gene')
        weights = available.float() / available.sum(1, keepdim=True).clamp_min(1)
        base = (weights[..., None] * sy).sum(1)
        return (qc, sc, sy, valid, weights, base)

    def _precision(self, available, weights, support_cov):
        b, k, g = available.shape
        if support_cov is None or support_cov.shape != (b, k, k, g, 2):
            raise ValueError('Program arms require full support_cov B,K,K,G,2, including shared endpoints')
        mask = available[:, :, None, :, None] & available[:, None, :, :, None]
        cov = torch.where(mask, support_cov.float(), torch.zeros_like(support_cov).float())
        if not torch.isfinite(cov).all():
            raise ValueError('Nonfinite measured support covariance')
        diagonal = cov.diagonal(dim1=1, dim2=2).permute(0, 3, 1, 2)
        if (diagonal < -1e-07).any():
            raise ValueError('Negative support sampling variance')
        row = torch.einsum('bklgc,blg->bkgc', cov, weights)
        pooled = (weights[..., None] * row).sum(1).clamp_min(0)
        independent = (weights.square()[..., None] * diagonal).sum(1).clamp_min(0)
        scale2 = self.response_units.square()
        count = available.sum(1).float().log1p()[..., None] / math.log(4)
        summary = torch.cat(((pooled / scale2).log1p(), (independent / scale2).log1p(), _signed_log((pooled - independent) / scale2), count), -1)
        pair = torch.cat(((diagonal.clamp_min(0) / scale2).log1p(), _signed_log(row / scale2), (pooled / scale2).log1p()[:, None].expand(-1, k, -1, -1), count[:, None].expand(-1, k, -1, -1)), -1)
        return (pair, summary, pooled, independent)

    def _pool(self, values, valid):
        member = self.program_membership[None] * valid[..., None]
        count = member.sum(1)
        pooled = torch.einsum('bgp,bgd->bpd', member, values) / count.clamp_min(1)[..., None]
        return (pooled, count > 0, count)

    def _lift(self, programs, program_valid):
        member = self.program_membership[None] * program_valid[:, None]
        return torch.einsum('bgp,bpd->bgd', member, programs) / member.sum(-1).clamp_min(1)[..., None]

    def _context(self, gene, control, valid):
        token = self.context_encoder(torch.cat((gene, control), -1))
        pooled, observed, _ = self._pool(token, valid)
        return (self._lift(pooled, observed), _mean_genes(pooled, observed), _mean_genes(token, valid), pooled)

    def forward(self, ids, qc, sc, sy, available, context=None, support_var=None, support_cov=None, return_explanation=False, context_columns=None, output_positions=None):
        if context is not None and context_columns is not None:
            raise ValueError('Specify context or context_columns, not both')
        qc, sc, sy, valid, weights, base = self._clean(ids, qc, sc, sy, available)
        b, k, g = available.shape
        gene = self.parent.gene_features(ids)[None].expand(b, -1, -1)
        pair_noise, precision, pooled, independent = self._precision(available, weights, support_cov)
        yn = sy / self.response_units
        pair_condition = torch.cat((gene[:, None].expand(-1, k, -1, -1), pair_noise), -1)
        encoded = _anchored(self.parent.gene_encoder, pair_condition, torch.cat((yn, yn.abs()), -1))
        local = (weights[..., None] * encoded).sum(1)
        global_effect = _mean_genes(local, valid)
        condition = torch.cat((gene, precision), -1)
        gene_delta = self.response_units * _anchored(self.parent.gene_decoder, condition, torch.cat((local, global_effect[:, None].expand(-1, g, -1)), -1))
        gene_delta = gene_delta + self.parent.common_gain * base
        pmean, program_valid, program_count = self._pool(local, valid)
        identity = self.program_identity[None].expand(b, -1, -1)
        program_effect = _anchored(self.parent.program_encoder, identity, pmean)
        lifted = self._lift(program_effect, program_valid)
        global_program = _mean_genes(program_effect, program_valid)
        program_condition = torch.cat((gene, self.program_membership[None].expand(b, -1, -1), precision), -1)
        program_delta = self.response_units * _anchored(self.parent.program_decoder, program_condition, torch.cat((lifted, global_program[:, None].expand(-1, g, -1)), -1))
        context_delta = torch.zeros_like(base)
        query_program_control = support_program_control = None
        if self.arm != 'program_shared':
            qn = qc / self.control_units
            sn = (weights[..., None] * (sc / self.control_units)).sum(1)
            qlocal, qglobal, qall, query_program_control = self._context(gene, qn, valid)
            slocal, sglobal, sall, support_program_control = self._context(gene, sn, valid)
            qcond = torch.cat((condition, qn, qlocal, qglobal[:, None].expand(-1, g, -1), qall[:, None].expand(-1, g, -1)), -1)
            scond = torch.cat((condition, sn, slocal, sglobal[:, None].expand(-1, g, -1), sall[:, None].expand(-1, g, -1)), -1)
            effects = torch.cat((local, lifted, global_program[:, None].expand(-1, g, -1)), -1)
            context_delta = self.response_units * (_anchored(self.context_decoder, qcond, effects) - _anchored(self.context_decoder, scond, effects))
        gene_delta = torch.where(valid[..., None], gene_delta, torch.zeros_like(gene_delta))
        program_delta = torch.where(valid[..., None], program_delta, torch.zeros_like(program_delta))
        context_delta = torch.where(valid[..., None], context_delta, torch.zeros_like(context_delta))
        raw = base + gene_delta + program_delta + context_delta
        prediction = project_delta(raw, qc)
        prediction = torch.where(valid[..., None], prediction, torch.zeros_like(prediction))
        if not return_explanation:
            if output_positions is not None:
                return (prediction[:, output_positions], base[:, output_positions])
            return (prediction, base)
        adjustment = prediction - raw
        clamp = adjustment.abs() > 1e-07
        explanation = dict(base=base, weights=weights, parent_unclamped=base + gene_delta, gene_correction=gene_delta, program_correction=program_delta, context_correction=context_delta, local_correction=gene_delta + program_delta + context_delta, pooled_support_sampling_variance=pooled, diagonal_only_sampling_variance=independent, shared_sampling_covariance_contribution=pooled - independent, precision_features=precision, unclamped_prediction=raw, projection_adjustment=adjustment, clamp_mask=clamp, valid=valid, clamp_fraction=(clamp & valid[..., None]).sum((0, 1)).float() / valid.sum().clamp_min(1), program_observed_gene_count=program_count, program_effect=program_effect, query_program_control=query_program_control, support_program_control=support_program_control, full_context_gene_count=self.genes, baseline_kind=self.baseline_kind, precision_interpretation='finite-cell summary sampling; not biological-replicate uncertainty')
        if output_positions is not None:
            return self._select(prediction, explanation, output_positions)
        return (prediction, explanation)

    @staticmethod
    def _select(prediction, explanation, positions):
        gene_keys = ('base', 'parent_unclamped', 'gene_correction', 'program_correction', 'context_correction', 'local_correction', 'pooled_support_sampling_variance', 'diagonal_only_sampling_variance', 'shared_sampling_covariance_contribution', 'precision_features', 'unclamped_prediction', 'projection_adjustment', 'clamp_mask', 'valid', 'sampling_attenuation')
        explanation = dict(explanation)
        for key in gene_keys:
            if key in explanation:
                explanation[key] = explanation[key][:, positions]
        explanation['weights'] = explanation['weights'][:, :, positions]
        valid, clamp = (explanation['valid'], explanation['clamp_mask'])
        explanation['clamp_fraction'] = (clamp & valid[..., None]).sum((0, 1)).float() / valid.sum().clamp_min(1)
        return (prediction[:, positions], explanation)
