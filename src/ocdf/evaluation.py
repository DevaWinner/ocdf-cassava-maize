"""OCDF F6 evaluation helpers: metrics, abstention, calibration error and the group bootstrap."""
import numpy as np

def full_metrics(p, y, eval_labels, n_out, threshold=None, groups=None):
    """Evaluation metrics (docs/METHODS.md). `eval_labels`: class indices present in this evaluation set (others are 'not evaluated').
    Predictions keep all n_out outputs (never masked), so predicting an absent class counts as an error."""
    from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
    pred = p.argmax(1); L = list(eval_labels)
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=L, zero_division=0)
    out = {'n_images': int(len(y)), 'n_groups': int(len(set(groups))) if groups is not None else None,
           'accuracy': float(accuracy_score(y, pred)),
           'balanced_accuracy': float(np.mean([np.mean(pred[y == c] == c) for c in L])),
           'macro_f1': float(f1_score(y, pred, labels=L, average='macro', zero_division=0)),
           'per_class': {int(c): {'precision': float(a), 'recall': float(b), 'f1': float(d), 'support': int(s)}
                         for c, a, b, d, s in zip(L, pr, rc, f1, sup)},
           'confusion': confusion_matrix(y, pred, labels=list(range(n_out))).tolist(),
           'nll': float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))),
           'brier': float(np.mean(np.sum((p - np.eye(n_out)[y]) ** 2, axis=1))),
           'ece_15': ece15(p, y)}
    if threshold is not None:
        acc = p.max(1) >= threshold; n = int(acc.sum())
        out.update({'threshold': float(threshold), 'coverage': float(acc.mean()),
                    'selective_accuracy': float((pred[acc] == y[acc]).mean()) if n else None,
                    'selective_risk': float((pred[acc] != y[acc]).mean()) if n else None,
                    'correct_accepted_yield': float(((pred == y) & acc).mean()),
                    'coverage_per_class': {int(c): float(acc[y == c].mean()) for c in L}})
    return out

def ece15(p, y, bins=15):
    conf = p.max(1); correct = (p.argmax(1) == y).astype(float); e = 0.0
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)

def fast_accuracy(y, pred):
    return float(np.mean(y == pred))

def fast_macro_f1(y, pred, labels, n_out):
    """Macro-F1 over `labels` from a bincount confusion matrix; identical to sklearn f1_score(average='macro', zero_division=0)."""
    cm = np.bincount(y * n_out + pred, minlength=n_out * n_out).reshape(n_out, n_out)
    tp = np.diag(cm)[labels].astype(float); fp = cm[:, labels].sum(0) - tp; fn = cm[labels, :].sum(1) - tp
    denom = 2 * tp + fp + fn
    return float(np.mean(np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)))

def group_bootstrap(stat_fn, groups, n_boot=2000, seed=2026):
    """95% percentile interval of stat_fn(index_array), resampling whole groups with replacement (Efron & Tibshirani, 1994)."""
    groups = np.asarray(groups); uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind='stable'); lengths = np.bincount(inv); starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    rng = np.random.default_rng(seed); vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq)); lens = lengths[pick]
        offs = np.repeat(starts[pick] - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
        vals[b] = stat_fn(order[offs])
    return {'estimate': float(stat_fn(np.arange(len(groups)))), 'ci95': [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))],
            'n_boot': n_boot, 'n_groups': int(len(uniq))}

def reliability_bins(p, y, bins=15):
    conf = p.max(1); correct = (p.argmax(1) == y).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1); rows = []
    for b in range(bins):
        m = idx == b
        rows.append({'bin_low': b / bins, 'bin_high': (b + 1) / bins, 'n': int(m.sum()),
                     'mean_confidence': float(conf[m].mean()) if m.any() else None, 'accuracy': float(correct[m].mean()) if m.any() else None})
    return rows

def risk_coverage(p, y):
    order = np.argsort(-p.max(1)); correct = (p.argmax(1) == y)[order]
    k = np.arange(1, len(y) + 1)
    return k / len(y), 1 - np.cumsum(correct) / k
