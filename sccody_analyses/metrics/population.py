"""Population reconstruction and differential-change fidelity metrics."""
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from geomloss import SamplesLoss
from .edger import _compare_vectors
from .differential_expression import de_logfc_diagnostics


def direction_agreement(real, predicted, mask=None):
    finite = np.isfinite(real) & np.isfinite(predicted)
    if mask is not None:
        finite &= mask
    return float(np.mean(np.sign(real[finite]) == np.sign(predicted[finite]))) if finite.any() else None


def shared_reference_metrics(predicted, observed, reference_mean, *, edge_table=None, genes=None):
    """Expression mean-logFC and real-count edgeR vector comparisons.

    Expression logFC uses processed expression means, independently of the
    donor-count edgeR logFC. Generated expression is never passed to edgeR.
    """
    diag = de_logfc_diagnostics(predicted, observed, np.asarray(reference_mean)[None, :])
    result = {key: value for key, value in diag.items() if not isinstance(value, np.ndarray)}
    real, pred = diag["logfc_real"], diag["logfc_pred"]
    result["direction_agreement_all"] = direction_agreement(real, pred)
    result["direction_agreement_nonzero"] = direction_agreement(real, pred, (real != 0) & (pred != 0))
    if edge_table is not None:
        table = edge_table.set_index("gene").reindex(list(map(str, genes)))
        edge = table.edgeR_logFC.to_numpy(dtype=float)
        fdr = table.edgeR_FDR.to_numpy(dtype=float)
        for label, vector in [("normalized_pred_vs_edgeR_real", pred), ("normalized_real_vs_edgeR_real", real)]:
            result.update({f"{label}_{k}": v for k,v in _compare_vectors(vector, edge, rank=edge).items()})
        for label, mask in [("true_fdr_lt_0p05", np.isfinite(fdr) & (fdr < .05)),
                            ("true_abs_log2fc_gt_0p25", np.isfinite(edge) & (np.abs(edge) > .25)),
                            ("true_abs_log2fc_gt_0p5", np.isfinite(edge) & (np.abs(edge) > .5))]:
            result[f"sign_agreement_{label}"] = direction_agreement(real, pred, mask)
            result[f"sign_filter_n_{label}"] = int(mask.sum())
    return result, diag


def distribution_metrics(observed, predicted, *, device="cpu", max_sinkhorn_cells=2000, blur=.05, scaling=.5):
    """Full-support per-gene Wasserstein/MMD and bounded gene Sinkhorn."""
    real, pred = np.asarray(observed, dtype=np.float32), np.asarray(predicted, dtype=np.float32)
    if real.ndim != 2 or pred.ndim != 2 or real.shape[1] != pred.shape[1] or min(len(real),len(pred)) < 2:
        raise ValueError("Distribution metrics need at least two cells and aligned genes")
    if not np.isfinite(real).all() or not np.isfinite(pred).all():
        raise ValueError("Distribution metrics require finite expression")
    means = _compare_vectors(pred.mean(axis=0), real.mean(axis=0), rank=real.mean(axis=0))
    distances = np.array([wasserstein_distance(real[:,g],pred[:,g]) for g in range(real.shape[1])])
    pairwise = cdist(real[:500],real[:500],metric="sqeuclidean")
    positive = pairwise[pairwise > 0]
    gamma = 1. / (float(np.median(positive))+1e-6) if len(positive) else 1.
    with torch.no_grad():
        x,y = torch.as_tensor(real,device=device),torch.as_tensor(pred,device=device)
        def kernel_sum(a,b,same=False):
            total = 0.
            b_norm = b.square().sum(dim=1)
            for start in range(0,len(a),256):
                z = a[start:start+256]
                squared = z.square().sum(dim=1)[:,None]+b_norm[None,:]-2*z@b.T
                kernel = torch.exp((-gamma*squared).double())
                if same:
                    indices = torch.arange(len(z),device=device)
                    kernel[indices,indices+start] = 0.
                total += float(kernel.sum())
            return total
        n,m=len(x),len(y)
        mmd = max(kernel_sum(x,x,True)/(n*(n-1))+kernel_sum(y,y,True)/(m*(m-1))-2*kernel_sum(x,y)/(n*m),0.)
        size=min(n,m,max_sinkhorn_cells)
        # Same deterministic bounded support policy as the research metric.
        sx,sy=x[:min(n,m)],y[:min(n,m)]
        if len(sx)>size:
            indices=torch.as_tensor(np.linspace(0,len(sx)-1,size,dtype=np.int64),device=device)
            sx,sy=sx[indices],sy[indices]
        if torch.equal(sx.amin(0),sx.amax(0)) and torch.equal(sy.amin(0),sy.amax(0)):
            sinkhorn=float(.5*(sx.mean(0)-sy.mean(0)).square().sum())
        else:
            sinkhorn=float(SamplesLoss("sinkhorn",p=2,blur=blur,scaling=scaling,backend="tensorized")(sx,sy))
    return {"mean_expression_corr":means["pearson_all"],"mean_expression_spearman":means["spearman_all"],
            "mean_expression_rmse":float(np.sqrt(np.mean((pred.mean(0)-real.mean(0))**2))),
            "mean_expression_mae":float(np.mean(np.abs(pred.mean(0)-real.mean(0)))),
            "per_gene_wasserstein_mean":float(distances.mean()),"per_gene_wasserstein_median":float(np.median(distances)),
            "mmd":mmd,"mmd_gamma":gamma,"gene_space_sinkhorn_at_chrono":sinkhorn,
            "gene_space_sinkhorn_n_cells":size,"n_real_cells":n,"n_pred_cells":m},distances


def edge_reference_metrics(predicted_logfc, observed_logfc):
    """Figure 2 comparisons against the observed donor-pseudobulk edgeR vector."""
    predicted, observed = np.asarray(predicted_logfc), np.asarray(observed_logfc)
    compared = _compare_vectors(predicted, observed, rank=observed)
    return {
        'de_vs_reference_logfc_pearson_all': compared['pearson_all'],
        'de_vs_reference_logfc_spearman_all': compared['spearman_all'],
        'de_vs_reference_top100_overlap_frac': compared['top100_overlap_frac'],
        'de_vs_reference_top200_overlap_frac': compared['top200_overlap_frac'],
        'de_direction_agreement_all': direction_agreement(observed, predicted),
        'de_direction_agreement_nonzero': direction_agreement(observed, predicted, (observed != 0) & (predicted != 0)),
        'sign_agreement_true_abs_log2fc_gt_0p25': direction_agreement(observed, predicted, abs(observed) > .25),
        'sign_agreement_true_abs_log2fc_gt_0p5': direction_agreement(observed, predicted, abs(observed) > .5),
    }
