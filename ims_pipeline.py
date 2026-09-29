#!/usr/bin/env python3
"""
IMS Spectral Discriminant Analysis — end-to-end reproducible pipeline
======================================================================

Runs the full workflow from PNG spectral images:

  1. Image digitization   pixel coordinates -> (drift time, intensity) curve
  2. Feature extraction   intensities at the 20 screened peaks P1-P20
  3. Statistical testing  Kruskal-Wallis + Dunn's post-hoc (Bonferroni)
  4. Dimensionality       PCA (top-12 discriminative features) + 95% ellipses
  5. Discriminant analysis  LDA, repeated 5-fold CV (10 repeats), mean±SD
  6. Logistic regression  one-vs-rest coefficient heatmap (interpretability)
  7. Decision curve       DCA (Net Benefit)

Usage:
    # Five groups (default)
    python ims_pipeline.py --data_dir <PNG_dir> --output_dir <out_dir>

    # Three groups (RA/OA/HC)
    python ims_pipeline.py --data_dir <PNG_dir> --output_dir <out_dir> --groups 3

    # Custom groups (comma-separated)
    python ims_pipeline.py --data_dir <PNG_dir> --output_dir <out_dir> --groups RA,OA,HC

Data: the directory must contain RA-*.png / OA-*.png / HC-*.png / SLE-*.png /
SS-*.png. RA/OA/HC are 945 px wide, SLE/SS are 866 px wide; the plot region is
auto-detected for non-standard sizes.

Dependencies: see requirements.txt
"""
import argparse
import os
import glob
import sys
import warnings

# Keep non-ASCII (e.g. paths) safe on Windows consoles regardless of code page.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (confusion_matrix, roc_curve, auc, accuracy_score,
                             f1_score)

try:
    import scikit_posthocs as sp  # Dunn's post-hoc (optional)
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

warnings.filterwarnings('ignore')
np.random.seed(42)

# ----------------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------------
# P1-P20: the 20 characteristic peaks (drift times, ms). These are NOT
# pre-specified — they were screened out of the full spectrum by between-group
# significance testing (the drift-time regions with significant group
# differences), yielding the 20 most discriminative peaks.
SIGNIFICANT_DT = np.array([
    5.58, 5.91, 6.35, 6.82, 7.53, 7.91, 9.45, 10.27, 11.15,
    12.14, 12.50, 14.02, 14.38, 14.80, 15.26, 16.00, 17.21,
    18.86, 19.24, 21.48,
])

# Colors for the five groups (three-group mode only uses the first three).
GROUP_COLORS = {
    'RA': '#4A90D9', 'OA': '#F5A623', 'HC': '#50B86C',
    'SLE': '#E91E63', 'SS': '#9C27B0',
}
ALL_GROUPS = ['RA', 'OA', 'HC', 'SLE', 'SS']

# Standard plot-region pixel coordinates for RA/OA/HC (945 px wide).
DEFAULT_X_MIN, DEFAULT_X_MAX = 91, 925
DEFAULT_Y_MIN, DEFAULT_Y_MAX = 38, 320
X_VALUE_RANGE = 22.0  # x-axis spans 0-22 ms

N_REPEATS = 10  # number of repeats for repeated CV
METRIC_NAMES = ['auc', 'sensitivity', 'specificity', 'ppv', 'npv', 'accuracy', 'f1']


# ----------------------------------------------------------------------------
# Step 1: image digitization
# ----------------------------------------------------------------------------
def detect_plot_bounds(image_path):
    """Auto-detect the four plot-region boundaries for non-standard image sizes.

    The axes are long dark straight lines, so scan inward from each edge and
    locate the row/column where >60% of pixels are dark.
    """
    img = Image.open(image_path)
    gray = np.mean(np.array(img)[:, :, :3], axis=2)
    h, w = gray.shape

    y_bottom = h - 1
    for y in range(h - 1, h // 3, -1):
        if np.sum(gray[y, :] < 150) > w * 0.6:
            y_bottom = y
            break

    x_left = 0
    for x in range(0, w // 3):
        if np.sum(gray[:y_bottom, x] < 150) > y_bottom * 0.6:
            x_left = x
            break

    x_right = w - 1
    for x in range(w - 1, w // 2, -1):
        if np.sum(gray[:y_bottom, x] < 150) > y_bottom * 0.6:
            x_right = x
            break

    y_top = 0
    for y in range(0, y_bottom):
        if np.sum(gray[y, x_left:x_right] > 240) > (x_right - x_left) * 0.7:
            y_top = y
            break

    img.close()
    return x_left, x_right, y_top, y_bottom


def extract_spectrum(image_path):
    """Digitize one PNG spectrum into a (drift time, intensity) point set.

    In each pixel column the darkest pixel is the curve; its coordinates are
    converted to drift time (ms) and intensity (normalized to 0-1).
    """
    try:
        img = Image.open(image_path)
        gray = np.mean(np.array(img)[:, :, :3], axis=2)
        h, w = gray.shape

        if w >= 940:
            x_min, x_max, y_min, y_max = DEFAULT_X_MIN, DEFAULT_X_MAX, DEFAULT_Y_MIN, DEFAULT_Y_MAX
        else:
            x_min, x_max, y_min, y_max = detect_plot_bounds(image_path)

        x_scale = X_VALUE_RANGE / (x_max - x_min)
        spectrum = []
        for x in range(x_min, x_max + 1):
            col = gray[y_min:y_max + 1, x]
            min_idx = np.argmin(col)
            if col[min_idx] < 150:  # dark pixels are the curve
                y_px = min_idx + y_min
                dt = (x - x_min) * x_scale
                intensity = (y_max - y_px) / (y_max - y_min)
                spectrum.append((dt, intensity))
        img.close()
        if len(spectrum) < 50:
            return None
        return np.array(spectrum)
    except Exception as e:
        print(f'  ERROR {os.path.basename(image_path)}: {e}')
        return None


def extract_intensity_at_points(spec, target_dt):
    """Return intensities at given drift times via linear interpolation."""
    if spec is None or len(spec) < 5:
        return np.full(len(target_dt), np.nan)
    dt, intensity = spec[:, 0], spec[:, 1]
    _, idx = np.unique(dt, return_index=True)
    dt_u, int_u = dt[np.sort(idx)], intensity[np.sort(idx)]
    return np.array([np.interp(t, dt_u, int_u, left=np.nan, right=np.nan) for t in target_dt])


def load_all(data_dir, group_order):
    """Scan PNGs of the requested groups; return features X, labels y, raw spectra."""
    groups_raw = {g: [] for g in group_order}
    X, y = [], []
    for group in group_order:
        files = sorted(glob.glob(os.path.join(data_dir, f'{group}-*.png')),
                       key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f).split('-')[1]))))
        for f in files:
            spec = extract_spectrum(f)
            if spec is None:
                continue
            groups_raw[group].append(spec)
            feats = extract_intensity_at_points(spec, SIGNIFICANT_DT)
            if not np.all(np.isnan(feats)):
                X.append(feats)
                y.append(group)
        print(f'  {group}: {len(files)} files -> {len(groups_raw[group])} valid')
    X = np.array(X)
    y = np.array(y)
    good = np.sum(np.isnan(X), axis=1) == 0
    X, y = X[good], y[good]
    print(f'  Total valid samples: {len(X)} ({ {g: int(np.sum(y == g)) for g in group_order} })')
    return X, y, groups_raw


# ----------------------------------------------------------------------------
# Step 3: statistical testing + box plots
# ----------------------------------------------------------------------------
def sig_stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))


def plot_boxplots(X, y, out_dir, group_order):
    """Box plots of 20 features across groups, with K-W + Dunn's post-hoc (RA vs others)."""
    n_rows = (len(SIGNIFICANT_DT) + 4) // 5
    fig, axes = plt.subplots(n_rows, 5, figsize=(22, 4.5 * n_rows))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes[:len(SIGNIFICANT_DT)]):
        dt_val = SIGNIFICANT_DT[i]
        box_data = [X[y == g, i] for g in group_order]
        bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                        medianprops={'color': 'black', 'linewidth': 1.2})
        for patch, g in zip(bp['boxes'], group_order):
            patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.7)

        kw_p = stats.kruskal(*box_data).pvalue
        if kw_p < 0.05 and HAS_POSTHOCS:
            dunn = sp.posthoc_dunn(box_data, p_adjust='bonferroni')
            y_max = max(d.max() for d in box_data)
            y_range = y_max - min(d.min() for d in box_data)
            for si, j in enumerate(range(2, len(group_order) + 1)):  # RA(1) vs others
                p = dunn.iloc[0, j - 1]
                yy = y_max + y_range * 0.05 + si * y_range * 0.08
                ax.plot([1, j], [yy, yy], 'k-', linewidth=0.6)
                ax.text((1 + j) / 2, yy, sig_stars(p), ha='center', va='bottom',
                        fontsize=7, fontweight='bold')
        ax.set_title(f'{dt_val:.2f} ms', fontsize=9, fontweight='bold')
        ax.set_xticklabels(group_order, fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
    for ax in axes[len(SIGNIFICANT_DT):]:
        ax.set_visible(False)
    fig.suptitle('P1-P20 feature intensities across groups (K-W + Dunn post-hoc)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ----------------------------------------------------------------------------
# Step 4: PCA
# ----------------------------------------------------------------------------
def plot_pca(X, y, out_dir, group_order):
    """PCA on the top-12 K-W-ranked features; score plot with 95% confidence ellipses."""
    kw = [stats.kruskal(*[X[y == g, i] for g in group_order]).statistic for i in range(20)]
    top_idx = sorted(np.argsort(kw)[-12:])
    X_top = StandardScaler().fit_transform(X[:, top_idx])
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_top)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))
    for g in group_order:
        m = y == g
        ax.scatter(X_pca[m, 0], X_pca[m, 1], c=GROUP_COLORS[g], label=g,
                   alpha=0.8, s=65, edgecolors='none')
        if m.sum() > 2:
            cov = np.cov(X_pca[m, 0], X_pca[m, 1])
            eigv, eigvec = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
            w, h = 2 * np.sqrt(5.991 * eigv)  # 95% confidence
            ax.add_patch(Ellipse((X_pca[m, 0].mean(), X_pca[m, 1].mean()), w, h,
                                 angle=angle, facecolor=GROUP_COLORS[g], alpha=0.08,
                                 edgecolor=GROUP_COLORS[g], linewidth=2))
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)')
    ax.legend(framealpha=0.9)
    ax.grid(False)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pca_score_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return evr, top_idx


# ----------------------------------------------------------------------------
# Step 5-7: LDA discriminant analysis (repeated CV)
# ----------------------------------------------------------------------------
def calc_metrics(yt, yp, yprob):
    cm = confusion_matrix(yt, yp)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr, tpr, _ = roc_curve(yt, yprob)
    return {
        'auc': auc(fpr, tpr),
        'sensitivity': tp / (tp + fn) if tp + fn else 0,
        'specificity': tn / (tn + fp) if tn + fp else 0,
        'ppv': tp / (tp + fp) if tp + fp else 0,
        'npv': tn / (tn + fn) if tn + fn else 0,
        'accuracy': accuracy_score(yt, yp),
        'f1': f1_score(yt, yp),
    }


def dca_net_benefit(threshold, y_true, y_prob):
    n = len(y_true)
    tp = np.sum((y_prob >= threshold) & (y_true == 1))
    fp = np.sum((y_prob >= threshold) & (y_true == 0))
    return tp / n - (fp / n) * (threshold / (1 - threshold))


def analyze_pairwise(X, y, pos_class, neg_name, out_dir):
    """One binary comparison: RA vs a group (or non-RA). Returns mean±SD metrics."""
    y_bin = np.array([1 if l == pos_class else 0 for l in y])
    X_scaled = StandardScaler().fit_transform(X)

    X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y_bin, test_size=0.3,
                                                stratify=y_bin, random_state=42)
    lda = LinearDiscriminantAnalysis().fit(X_tr, y_tr)
    m_val = calc_metrics(y_val, lda.predict(X_val), lda.predict_proba(X_val)[:, 1])

    # repeated 5-fold CV: per-fold metrics -> mean ± SD
    fold = {k: [] for k in METRIC_NAMES}
    for rep in range(N_REPEATS):
        cv = StratifiedKFold(5, shuffle=True, random_state=42 + rep)
        for tr, va in cv.split(X_scaled, y_bin):
            m = LinearDiscriminantAnalysis().fit(X_scaled[tr], y_bin[tr])
            yp, yprob = m.predict(X_scaled[va]), m.predict_proba(X_scaled[va])[:, 1]
            r = calc_metrics(y_bin[va], yp, yprob)
            for k in METRIC_NAMES:
                fold[k].append(r[k])
    mean = {k: float(np.mean(fold[k])) for k in METRIC_NAMES}
    sd = {k: float(np.std(fold[k])) for k in METRIC_NAMES}

    # ROC (validation set)
    fpr, tpr, _ = roc_curve(y_val, lda.predict_proba(X_val)[:, 1])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, linewidth=2, label=f'{pos_class} vs {neg_name} (AUC={m_val["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')
    ax.set_title(f'ROC — {pos_class} vs {neg_name}')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'roc_{pos_class}_vs_{neg_name.replace(" ", "_")}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # DCA (pooled CV probabilities)
    y_prob_cv = cross_val_predict(lda, X_scaled, y_bin,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                  method='predict_proba')[:, 1]
    thr = np.linspace(0.01, 0.99, 99)
    nb_model = [dca_net_benefit(t, y_bin, y_prob_cv) for t in thr]
    nb_all = [dca_net_benefit(t, y_bin, np.ones(len(y_bin))) for t in thr]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(thr, nb_model, linewidth=2, label=f'{pos_class} vs {neg_name}')
    ax.plot(thr, nb_all, 'k--', linewidth=1.5, label='Treat All')
    ax.plot(thr, np.zeros(99), 'k-', linewidth=1, label='Treat None')
    ax.set_xlabel('Threshold Probability'); ax.set_ylabel('Net Benefit')
    ax.set_title(f'DCA — {pos_class} vs {neg_name}')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim([0, 1])
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f'dca_{pos_class}_vs_{neg_name.replace(" ", "_")}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return {'mean': mean, 'sd': sd, 'val': m_val}


# ----------------------------------------------------------------------------
# Step 6: logistic regression coefficient heatmap (one-vs-rest, any group count)
# ----------------------------------------------------------------------------
def plot_logreg_coef(X, y, out_dir, group_order):
    """One-vs-rest logistic regression coefficient heatmap (which peak matters most)."""
    Xs = StandardScaler().fit_transform(X)
    n_groups = len(group_order)
    coef = np.zeros((n_groups, 20))
    for gi, g in enumerate(group_order):
        yb = np.where(y == g, 1, 0)
        lr = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42).fit(Xs, yb)
        coef[gi] = lr.coef_[0]
    fig, ax = plt.subplots(figsize=(18, 1.5 + 1.2 * n_groups))
    im = ax.imshow(coef, aspect='auto', cmap='Blues', interpolation='nearest')
    ax.set_xticks(range(20))
    ax.set_xticklabels([f'p{i+1}' for i in range(20)], fontstyle='italic', rotation=45, ha='right')
    ax.set_yticks(range(n_groups)); ax.set_yticklabels(group_order, fontweight='bold')
    vmin, vmax = coef.min(), coef.max()
    for ri in range(n_groups):
        for ci in range(20):
            v = coef[ri, ci]
            ax.text(ci, ri, f'{v:.2f}', ha='center', va='center', fontsize=8.5,
                    fontweight='bold', color='white' if (v - vmin) / (vmax - vmin + 1e-10) > 0.5 else '#0D2137')
    plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    ax.set_title('Logistic Regression Coefficients (one-vs-rest)')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'logreg_coefficients.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return coef


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='IMS spectral discriminant analysis reproducible pipeline')
    ap.add_argument('--data_dir', default='.', help='Directory containing PNG spectra')
    ap.add_argument('--output_dir', default='output', help='Output directory')
    ap.add_argument('--groups', default='5',
                    help='Groups: 3 (=RA/OA/HC), 5 (=RA/OA/HC/SLE/SS), or comma-separated names')
    args = ap.parse_args()

    # Resolve groups
    if args.groups == '3':
        group_order = ['RA', 'OA', 'HC']
    elif args.groups == '5':
        group_order = list(ALL_GROUPS)
    else:
        group_order = [g.strip() for g in args.groups.split(',') if g.strip()]

    os.makedirs(args.output_dir, exist_ok=True)
    print('=' * 60)
    print('IMS Spectral Discriminant Analysis Pipeline')
    print(f'Groups: {group_order}')
    print(f'Data dir: {args.data_dir}   Output dir: {args.output_dir}')
    print('=' * 60)

    print('\n[Step 1-2] Image digitization + feature extraction ...')
    X, y, groups_raw = load_all(args.data_dir, group_order)

    print('\n[Step 3] Group box plots (K-W + Dunn) ...')
    plot_boxplots(X, y, args.output_dir, group_order)

    print('\n[Step 4] PCA (top-12 discriminative features) ...')
    evr, top_idx = plot_pca(X, y, args.output_dir, group_order)
    print(f'  PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}%, PC3={evr[2]*100:.1f}%')

    print('\n[Step 5-7] LDA discriminant analysis (repeated 5-fold CV, mean±SD) ...')
    neg_groups = [g for g in group_order if g != 'RA']
    comparisons = [(g, g) for g in neg_groups] + [('non-RA', 'non-RA')]
    results = {}
    for neg, comp in comparisons:
        if neg == 'non-RA':
            yy = np.array(['RA' if l == 'RA' else 'non-RA' for l in y])
            XX = X
        else:
            m = (y == 'RA') | (y == neg)
            XX, yy = X[m], y[m]
        results[f'RA vs {neg}'] = analyze_pairwise(XX, yy, 'RA', neg, args.output_dir)

    # Summary table
    rows = [['Comparison'] + METRIC_NAMES]
    for comp, r in results.items():
        m, sd = r['mean'], r['sd']
        rows.append([comp] + [f'{m[k]:.4f}±{sd[k]:.4f}' for k in METRIC_NAMES])
    with open(os.path.join(args.output_dir, 'discriminant_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f'LDA discriminant analysis — repeated 5-fold CV (mean±SD, {N_REPEATS} repeats)\n')
        f.write('=' * 130 + '\n')
        for row in rows:
            f.write('  '.join(f'{c:>22s}' if i > 0 else f'{c:<12s}' for i, c in enumerate(row)) + '\n')
    print('\n' + '=' * 130)
    for row in rows:
        print('  '.join(f'{c:>22s}' if i > 0 else f'{c:<12s}' for i, c in enumerate(row)))
    print('=' * 130)

    print('\n[Step 6] Logistic regression coefficient heatmap ...')
    plot_logreg_coef(X, y, args.output_dir, group_order)

    print('\nDone! Results saved to', os.path.abspath(args.output_dir))


if __name__ == '__main__':
    main()
