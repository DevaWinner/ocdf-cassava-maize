# ---- OCDF F4 calibration, abstention and selection helpers (Notebook 04) ----
import numpy as np

def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def nll(z, y, T=1.0):
    p = softmax(z, T)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))

def fit_temperature(z, y, lo=0.05, hi=20.0):
    """Calibration: positive T minimising negative log-likelihood on the calibration partition (Guo et al., 2017).
    Golden-section search on log T (NLL is unimodal in T for fixed logits)."""
    a, b = np.log(lo), np.log(hi); g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(z, y, np.exp(c)), nll(z, y, np.exp(d))
    for _ in range(80):
        if fc < fd: b, d, fd = d, c, fc; c = b - g * (b - a); fc = nll(z, y, np.exp(c))
        else: a, c, fc = c, d, fd; d = a + g * (b - a); fd = nll(z, y, np.exp(d))
    T = float(np.exp((a + b) / 2))
    return {'T': round(T, 4), 'nll_before': round(nll(z, y, 1.0), 4), 'nll_after': round(nll(z, y, T), 4)}

def brier(p, y):
    oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))

def ece(p, y, bins=15):
    conf = p.max(1); pred = p.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[0]) & (conf <= edges[1])
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)

def threshold_table(p, y, groups, thresholds=None):
    """Coverage, selective accuracy and accepted groups for each threshold (see docs/METHODS.md)."""
    thresholds = np.round(np.arange(0.50, 0.995, 0.01), 2) if thresholds is None else thresholds
    conf = p.max(1); pred = p.argmax(1); groups = np.asarray(groups); rows = []
    for t in thresholds:
        acc_mask = conf >= t; n_acc = int(acc_mask.sum())
        rows.append({'threshold': float(t), 'coverage': n_acc / len(y),
                     'selective_accuracy': float((pred[acc_mask] == y[acc_mask]).mean()) if n_acc else None,
                     'selective_risk': float((pred[acc_mask] != y[acc_mask]).mean()) if n_acc else None,
                     'correct_accepted_yield': float(((pred == y) & acc_mask).mean()),
                     'accepted_groups': int(len(set(groups[acc_mask]))), 'accepted': n_acc})
    return rows

def choose_threshold(rows, min_acc=0.95, min_cov=0.50, min_groups=50, fallback=0.90):
    """Highest-coverage threshold with selective accuracy >= 0.95, coverage >= 0.50 and >= 50 accepted groups;
    otherwise 'target infeasible' and the predeclared 0.90 for description only."""
    ok = [r for r in rows if r['selective_accuracy'] is not None and r['selective_accuracy'] >= min_acc
          and r['coverage'] >= min_cov and r['accepted_groups'] >= min_groups]
    if ok:
        best = max(ok, key=lambda r: (r['coverage'], -r['threshold']))
        return {'threshold': best['threshold'], 'status': 'target met', **{k: best[k] for k in ('coverage', 'selective_accuracy', 'accepted_groups')}}
    fb = min(rows, key=lambda r: abs(r['threshold'] - fallback))
    return {'threshold': fallback, 'status': 'target infeasible (0.90 used for description only)',
            **{k: fb[k] for k in ('coverage', 'selective_accuracy', 'accepted_groups')}}

def per_class_coverage(p, y, t, classes):
    conf = p.max(1); out = {}
    for i, c in enumerate(classes):
        m = y == i
        out[c] = round(float((conf[m] >= t).mean()), 4) if m.any() else None
    return out

def select_architecture(agg_rows, size_bytes=None, margin=0.01):
    """Pre-declared selection rule: highest mean selection macro-F1; within `margin` of the best, the smaller converted file."""
    best = max(agg_rows, key=lambda r: r['mean_macro_f1'])
    close = [r for r in agg_rows if best['mean_macro_f1'] - r['mean_macro_f1'] <= margin + 1e-12]
    if len(close) > 1 and size_bytes:
        pick = min(close, key=lambda r: size_bytes[r['architecture']])
        return pick['architecture'], f'within {margin} of best; smallest FP32 file chosen'
    return best['architecture'], (f"highest mean selection macro-F1 ({best['mean_macro_f1']:.4f}); "
                                   f"no other architecture within {margin}")
