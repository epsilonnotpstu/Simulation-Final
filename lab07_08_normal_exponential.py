"""
=============================================================
CIT 324 - Simulation and Modeling Sessional
Lab 07: Normal Distribution — Unimodal & Multimodal, Blood Pressure
Lab 08: Exponential Distribution — COVID Waves, Multi-rate plot
=============================================================
Theory (Averill M. Law, Chapter 8):
  - Normal: Symmetric, bell-shaped, parameterized by (μ, σ)
  - Multimodal: mixture of Normals → multiple peaks
  - Exponential: memoryless distribution for inter-event times
    f(x) = λ e^(-λx), CDF: F(x) = 1 - e^(-λx)
    P(X > t) = e^(-λt)  ← key formula for Lab 08
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from tabulate import tabulate


# ═══════════════════════════════════════════════════════════
# LAB 07 — NORMAL DISTRIBUTION
# ═══════════════════════════════════════════════════════════

def lab07_normal(mu1=100, sigma1=20, n_samples=200,
                 bp_mu=80, bp_sigma=20, seed=42):
    """
    Shows:
    1. Unimodal normal: N(100, 20)  — sample size 200
    2. Multimodal: mixture of 3 normals
    3. Blood pressure distribution: N(80, 20)
    """
    rng = np.random.default_rng(seed)

    print("\n" + "="*62)
    print("   LAB 07: NORMAL DISTRIBUTION")
    print("="*62)

    # ── Generate samples ──
    sample_uni = rng.normal(mu1, sigma1, n_samples)

    # Multimodal = mixture of 3 Gaussians
    mix_mu    = [-3, 0, 4]
    mix_sigma = [0.8, 1.2, 0.6]
    mix_w     = [0.35, 0.40, 0.25]
    sample_multi = np.concatenate([
        rng.normal(m, s, int(w * 1000))
        for m, s, w in zip(mix_mu, mix_sigma, mix_w)
    ])
    # Blood pressure
    sample_bp = rng.normal(bp_mu, bp_sigma, 5000)

    # ── Stats for unimodal sample ──
    print("\n  ── Unimodal Sample Statistics (N=200, μ=100, σ=20) ──")
    rows = [
        ["Sample Mean",      f"{np.mean(sample_uni):.4f}", f"{mu1}"],
        ["Sample Std Dev",   f"{np.std(sample_uni):.4f}",  f"{sigma1}"],
        ["Sample Variance",  f"{np.var(sample_uni):.4f}",  f"{sigma1**2}"],
        ["Skewness",         f"{stats.skew(sample_uni):.4f}", "0 (symmetric)"],
        ["Kurtosis",         f"{stats.kurtosis(sample_uni):.4f}", "0 (normal)"],
    ]
    print(tabulate(rows, headers=["Metric", "Simulated", "Theoretical"],
                   tablefmt="rounded_grid"))

    # ── Blood pressure probabilities ──
    rv_bp = stats.norm(bp_mu, bp_sigma)
    print("\n  ── Blood Pressure Probabilities (μ=80, σ=20) ──")
    bp_rows = [
        ["P(BP < 60)", f"{rv_bp.cdf(60):.4f}"],
        ["P(60 ≤ BP ≤ 100)", f"{rv_bp.cdf(100)-rv_bp.cdf(60):.4f}"],
        ["P(BP > 100)", f"{rv_bp.sf(100):.4f}"],
        ["P(BP > 120) — Hypertension?", f"{rv_bp.sf(120):.4f}"],
    ]
    print(tabulate(bp_rows, headers=["Event", "Probability"],
                   tablefmt="rounded_grid"))

    # ── PLOT ──
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("Lab 07 — Normal Distribution",
                 fontsize=14, fontweight='bold', color='#2c3e50')

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ─ A: Unimodal histogram + PDF ─
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.hist(sample_uni, bins=20, density=True, color='#3498db',
             alpha=0.7, edgecolor='white', label='Sample (N=200)')
    x = np.linspace(mu1 - 4*sigma1, mu1 + 4*sigma1, 300)
    ax1.plot(x, stats.norm.pdf(x, mu1, sigma1), 'r-',
             linewidth=2.5, label=f'Normal PDF N({mu1},{sigma1})')
    ax1.axvline(mu1, color='green', linestyle='--', linewidth=2, label=f'μ={mu1}')
    ax1.axvline(mu1 - sigma1, color='orange', linestyle=':', linewidth=1.5,
                label=f'μ±σ ({mu1-sigma1}, {mu1+sigma1})')
    ax1.axvline(mu1 + sigma1, color='orange', linestyle=':', linewidth=1.5)
    ax1.set_title('Unimodal Normal: N(100, 20) — n=200', fontweight='bold')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # ─ B: Q-Q plot ─
    ax2 = fig.add_subplot(gs[0, 2])
    (osm, osr), (slope, intercept, r) = stats.probplot(sample_uni, dist="norm")
    ax2.scatter(osm, osr, color='#3498db', s=20, alpha=0.7)
    ax2.plot(osm, slope*np.array(osm) + intercept, 'r-', linewidth=2)
    ax2.set_title(f'Q-Q Plot  (R²={r**2:.4f})', fontweight='bold')
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # ─ C: Multimodal ─
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.hist(sample_multi, bins=60, density=True, color='#8e44ad',
             alpha=0.7, edgecolor='white', label='Mixture Sample')
    xm = np.linspace(-7, 8, 500)
    mixture_pdf = sum(w * stats.norm.pdf(xm, m, s)
                      for w, m, s in zip(mix_w, mix_mu, mix_sigma))
    ax3.plot(xm, mixture_pdf, 'r-', linewidth=2.5, label='Mixture PDF')
    for w, m, s, c in zip(mix_w, mix_mu, mix_sigma,
                           ['#3498db','#e74c3c','#2ecc71']):
        ax3.plot(xm, w * stats.norm.pdf(xm, m, s),
                 '--', linewidth=1.5, color=c,
                 label=f'N({m},{s}) w={w}')
    ax3.set_title('Multimodal Distribution (Mixture of 3 Normals)',
                  fontweight='bold')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')

    # ─ D: Blood pressure ─
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(sample_bp, bins=50, density=True, color='#e74c3c',
             alpha=0.7, edgecolor='white', label='BP Sample (n=5000)')
    xbp = np.linspace(0, 160, 400)
    ax4.plot(xbp, stats.norm.pdf(xbp, bp_mu, bp_sigma),
             'b-', linewidth=2.5, label=f'N({bp_mu},{bp_sigma})')
    ax4.axvline(80, color='green', linestyle='--', linewidth=2, label='Normal BP=80')
    ax4.axvline(120, color='red', linestyle=':', linewidth=2, label='Hypertension=120')
    ax4.set_title('Diastolic Blood Pressure Distribution', fontweight='bold')
    ax4.set_xlabel('Blood Pressure (mmHg)')
    ax4.set_ylabel('Density')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')

    plt.savefig('/mnt/user-data/outputs/lab07_normal.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab07_normal.png")
    plt.show()


# ═══════════════════════════════════════════════════════════
# LAB 08 — EXPONENTIAL DISTRIBUTION
# ═══════════════════════════════════════════════════════════

def lab08_exponential(mean_wave=100, threshold=120,
                      rates=None, seed=42):
    """
    COVID wave inter-event time ~ Exponential(λ = 1/100 days).
    P(X > 120) = e^(-120/100)
    Also simulate multiple rate parameters.
    """
    if rates is None:
        rates = [0.5, 1.0, 2.0, 4.0]

    rng   = np.random.default_rng(seed)
    lam   = 1.0 / mean_wave  # rate parameter

    print("\n" + "="*62)
    print("   LAB 08: EXPONENTIAL — COVID-19 Wave Intervals")
    print("="*62)
    print(f"\n   Mean time between waves : {mean_wave} days")
    print(f"   Rate λ = 1/{mean_wave}  : {lam:.6f}")
    print(f"   Threshold               : {threshold} days")

    # ── P(X > threshold) = e^(-λ * threshold) ──
    P_gt = np.exp(-lam * threshold)
    # ── E[X] = 1/λ, Var(X) = 1/λ² ──
    E_X  = mean_wave
    V_X  = mean_wave**2

    print("\n  ── Key Computations ──")
    rows = [
        ["P(X > 120 days)",          f"{P_gt:.6f}  ≈ {P_gt*100:.2f}%"],
        ["P(X ≤ 120 days)",          f"{1-P_gt:.6f}"],
        ["E[X] = 1/λ",              f"{E_X} days"],
        ["Var(X) = 1/λ²",           f"{V_X} days²"],
        ["Std(X) = 1/λ",            f"{E_X} days"],
        ["Median = ln(2)/λ",        f"{np.log(2)*mean_wave:.2f} days"],
    ]
    print(tabulate(rows, headers=["Metric", "Value"],
                   tablefmt="rounded_grid"))

    # Simulate
    n_sim    = 10000
    sim_data = rng.exponential(mean_wave, n_sim)
    P_gt_sim = np.mean(sim_data > threshold)

    print(f"\n  ── Verification (Monte Carlo, n={n_sim}) ──")
    vrows = [
        ["P(X>120) Theoretical", f"{P_gt:.6f}"],
        ["P(X>120) Simulated",   f"{P_gt_sim:.6f}"],
        ["Absolute Error",       f"{abs(P_gt - P_gt_sim):.6f}"],
    ]
    print(tabulate(vrows, headers=["", "Value"], tablefmt="rounded_grid"))

    # ── Rate table ──
    print("\n  ── Exponential Distribution — Multiple Rates ──")
    rate_rows = [
        [r, f"{1/r:.4f}", f"{1/r:.4f}", f"{1/r**2:.4f}",
         f"{np.exp(-r*1):.4f}", f"{np.exp(-r*2):.4f}"]
        for r in rates
    ]
    print(tabulate(rate_rows,
                   headers=["λ", "Mean(1/λ)", "Median(ln2/λ)≈",
                             "Var(1/λ²)", "P(X>1)", "P(X>2)"],
                   tablefmt="rounded_grid"))

    # ── PLOT ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Lab 08 — Exponential Distribution",
                 fontsize=14, fontweight='bold', color='#2c3e50')

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    # ─ A: COVID wave distribution ─
    ax1 = axes[0, 0]
    x_cv = np.linspace(0, 500, 500)
    pdf  = lam * np.exp(-lam * x_cv)
    ax1.plot(x_cv, pdf, '#2c3e50', linewidth=2.5, label=f'Exp(λ=1/{mean_wave})')
    ax1.fill_between(x_cv, pdf, where=x_cv > threshold,
                     alpha=0.4, color='#e74c3c',
                     label=f'P(X>{threshold}) = {P_gt:.4f}')
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'x = {threshold} days')
    ax1.axvline(mean_wave, color='green', linestyle=':', linewidth=2,
                label=f'Mean = {mean_wave} days')
    ax1.set_title('COVID Wave: P(Next wave > 120 days)', fontweight='bold')
    ax1.set_xlabel('Days Between Waves')
    ax1.set_ylabel('PDF f(x)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # ─ B: Simulated histogram ─
    ax2 = axes[0, 1]
    ax2.hist(sim_data, bins=60, density=True, color='#3498db',
             alpha=0.7, edgecolor='white', label=f'Simulated (n={n_sim})')
    ax2.plot(x_cv, pdf, 'r-', linewidth=2.5, label='Theoretical PDF')
    ax2.axvline(threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Threshold={threshold}')
    ax2.set_title('Simulated vs Theoretical PDF', fontweight='bold')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # ─ C: Multiple rate PDFs ─
    ax3 = axes[1, 0]
    x_r = np.linspace(0, 8, 300)
    for r, c in zip(rates, colors):
        ax3.plot(x_r, r * np.exp(-r * x_r),
                 linewidth=2.5, color=c, label=f'λ={r}')
    ax3.set_title('Exponential PDF — Multiple Rate Parameters', fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('f(x)')
    ax3.legend()
    ax3.set_ylim(0, 4.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')

    # ─ D: Survival functions ─
    ax4 = axes[1, 1]
    for r, c in zip(rates, colors):
        ax4.plot(x_r, np.exp(-r * x_r),
                 linewidth=2.5, color=c, label=f'λ={r}')
    ax4.set_title('Survival Function P(X > x) — Multiple Rates', fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('P(X > x)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab08_exponential.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab08_exponential.png")
    plt.show()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  CIT 324 — Lab 07 & 08: Normal & Exponential    ║")
    print("╚══════════════════════════════════════════════════╝")
    print("\n  [7] Lab 07 — Normal Distribution")
    print("  [8] Lab 08 — Exponential Distribution")
    print("  [0] Run BOTH")

    try:
        ch = int(input("\n  Choice [0/7/8]: "))
    except ValueError:
        ch = 0

    if ch == 7 or ch == 0:
        try:
            mu_  = float(input("\n  Lab07 — Population Mean [e.g. 100]: "))
            sig_ = float(input("  Lab07 — Std Dev         [e.g.  20]: "))
            n_   = int(input("  Lab07 — Sample Size     [e.g. 200]: "))
        except ValueError:
            mu_, sig_, n_ = 100, 20, 200
        lab07_normal(mu1=mu_, sigma1=sig_, n_samples=n_)

    if ch == 8 or ch == 0:
        try:
            mw = float(input("\n  Lab08 — Mean days between waves [e.g. 100]: "))
            th = float(input("  Lab08 — Threshold to test P(X>?) [e.g. 120]: "))
        except ValueError:
            mw, th = 100, 120
        lab08_exponential(mean_wave=mw, threshold=th)

    print("\n✅ Labs 07-08 Complete!\n")


if __name__ == "__main__":
    main()
