"""
=============================================================
CIT 324 — Lab 12: Random Variates via Inverse Transform
         Lab 13: 5000 Exponential Variates via Inverse Transform
=============================================================
Theory (Averill M. Law, Chapter 8 — Generating Random Variates):

  Inverse Transform Method:
    If X has CDF F(x), then X = F^{-1}(U) where U ~ Uniform[0,1)
    This works because P(F^{-1}(U) ≤ x) = P(U ≤ F(x)) = F(x) ✓

  For Exponential(λ):
    F(x) = 1 - e^{-λx}
    F^{-1}(u) = -ln(1-u)/λ = -mean × ln(1-u)
    (Since 1-U ~ Uniform too, simplify to: X = -mean × ln(U))

  For Uniform(a, b):
    F^{-1}(u) = a + (b-a)u

  For Triangular(a, b, c):
    F^{-1}(u) = a + sqrt(u(b-a)(c-a))       if u ≤ (c-a)/(b-a)
              = b - sqrt((1-u)(b-a)(b-c))    otherwise

  For Weibull(k, λ):
    F^{-1}(u) = λ × (-ln(1-u))^{1/k}

  Lab 13 specifically: Generate 5000 Exp(λ=1) variates,
  plot histogram + overlay theoretical PDF.
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from tabulate import tabulate


# ── Inverse Transform Functions ──────────────────────────────

def inv_transform_exponential(u_arr, mean):
    """X = -mean * ln(U)"""
    return -mean * np.log(u_arr)


def inv_transform_uniform(u_arr, a, b):
    """X = a + (b-a)*U"""
    return a + (b - a) * u_arr


def inv_transform_triangular(u_arr, a, b, c):
    """Triangular(a=min, b=max, c=mode) via inverse transform"""
    fc = (c - a) / (b - a)
    result = np.where(
        u_arr <= fc,
        a + np.sqrt(u_arr * (b - a) * (c - a)),
        b - np.sqrt((1 - u_arr) * (b - a) * (b - c))
    )
    return result


def inv_transform_weibull(u_arr, shape, scale):
    """X = scale * (-ln(1-U))^{1/shape}"""
    return scale * (-np.log(1 - u_arr)) ** (1 / shape)


def inv_transform_normal_bm(u_arr, mu=0, sigma=1):
    """
    Box-Muller transform (uses pairs of uniforms):
    Z = sqrt(-2 ln U1) * cos(2π U2) → N(0,1)
    Then X = mu + sigma * Z
    """
    n     = len(u_arr) // 2
    U1    = u_arr[:n]
    U2    = u_arr[n:2*n]
    Z     = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    return mu + sigma * Z


# ── Generate U ~ Uniform[0,1) using LCG ─────────────────────

def lcg_generate(n, seed=42, a=16807, c=0, m=2**31 - 1):
    """Park & Miller LCG"""
    X   = seed
    out = np.empty(n)
    for i in range(n):
        X     = (a * X + c) % m
        out[i] = X / m
    return out


def goodness_of_fit(samples, dist_name, dist_obj, params):
    """KS test + AIC for one distribution (params already baked into frozen dist)."""
    # dist_obj is already a frozen rv object
    ks_stat, ks_p = stats.kstest(samples, dist_obj.cdf)
    log_l = np.sum(dist_obj.logpdf(samples))
    aic   = 2 * len(params) - 2 * log_l
    return ks_stat, ks_p, aic


# ═══════════════════════════════════════════════════════════
# LAB 12
# ═══════════════════════════════════════════════════════════
def lab12_inverse_transform():
    print("\n" + "="*65)
    print("   LAB 12: RANDOM VARIATES — INVERSE TRANSFORM METHOD")
    print("="*65)

    try:
        n = int(input("\n  Number of variates to generate [e.g. 5000]: "))
        seed = int(input("  LCG Seed [e.g. 42]: "))
    except ValueError:
        n, seed = 5000, 42

    # ── Generate base uniforms via LCG ──
    U = lcg_generate(n * 2, seed=seed)  # extra for Box-Muller

    print(f"\n  Generated {n} uniform base variates using LCG (Park & Miller)")

    # ── Distributions ──
    configs = [
        {
            'name'     : 'Exponential(mean=2)',
            'samples'  : inv_transform_exponential(U[:n], mean=2.0),
            'theory'   : stats.expon(scale=2.0),
            'pdf_range': (0, 10),
            'color'    : '#3498db',
            'params'   : (0, 2.0),
        },
        {
            'name'     : 'Uniform(a=2, b=8)',
            'samples'  : inv_transform_uniform(U[:n], a=2, b=8),
            'theory'   : stats.uniform(loc=2, scale=6),
            'pdf_range': (1.5, 8.5),
            'color'    : '#e74c3c',
            'params'   : (2, 6),
        },
        {
            'name'     : 'Triangular(a=0, b=10, c=4)',
            'samples'  : inv_transform_triangular(U[:n], a=0, b=10, c=4),
            'theory'   : stats.triang(c=0.4, loc=0, scale=10),
            'pdf_range': (-0.5, 10.5),
            'color'    : '#2ecc71',
            'params'   : (0.4, 0, 10),
        },
        {
            'name'     : 'Weibull(k=2, λ=3)',
            'samples'  : inv_transform_weibull(U[:n], shape=2, scale=3),
            'theory'   : stats.weibull_min(c=2, scale=3),
            'pdf_range': (0, 9),
            'color'    : '#f39c12',
            'params'   : (2, 0, 3),
        },
        {
            'name'     : 'Normal(μ=5, σ=1.5) via Box-Muller',
            'samples'  : inv_transform_normal_bm(U, mu=5, sigma=1.5),
            'theory'   : stats.norm(loc=5, scale=1.5),
            'pdf_range': (0, 10),
            'color'    : '#9b59b6',
            'params'   : (5, 1.5),
        },
    ]

    # ── Print statistics table ──
    print("\n  ── Generated Variates — Statistics vs Theory ──")
    stat_rows = []
    for cfg in configs:
        s = cfg['samples']
        t = cfg['theory']
        stat_rows.append([
            cfg['name'],
            f"{np.mean(s):.4f}",
            f"{t.mean():.4f}",
            f"{np.std(s):.4f}",
            f"{t.std():.4f}",
        ])
    print(tabulate(stat_rows,
                   headers=["Distribution", "Sim Mean", "Theo Mean",
                             "Sim Std", "Theo Std"],
                   tablefmt="rounded_grid"))

    # ── KS Test ──
    print("\n  ── KS Goodness-of-Fit Test ──")
    ks_rows = []
    for cfg in configs:
        ks, p, _ = goodness_of_fit(cfg['samples'], cfg['name'],
                                    cfg['theory'], cfg['params'])
        ks_rows.append([cfg['name'], f"{ks:.4f}", f"{p:.4f}",
                        "✅ PASS" if p > 0.05 else "❌ FAIL"])
    print(tabulate(ks_rows,
                   headers=["Distribution", "KS Stat", "p-value", "Result"],
                   tablefmt="rounded_grid"))

    # ── PLOTS ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Lab 12 — Inverse Transform Method: Multiple Distributions",
                 fontsize=14, fontweight='bold', color='#2c3e50')

    for i, (cfg, ax) in enumerate(zip(configs, axes.flat)):
        s     = cfg['samples']
        t     = cfg['theory']
        xr    = np.linspace(*cfg['pdf_range'], 300)
        ax.hist(s, bins=40, density=True,
                color=cfg['color'], alpha=0.7,
                edgecolor='white', label='Simulated')
        ax.plot(xr, t.pdf(xr), 'k-', linewidth=2.5,
                label='Theoretical PDF')
        ax.axvline(np.mean(s), color='red', linestyle='--',
                   linewidth=1.5, label=f'Mean={np.mean(s):.2f}')
        ax.set_title(cfg['name'], fontsize=9, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')

    # Last subplot: Q-Q for exponential
    ax_qq = axes.flat[5]
    exp_samples = configs[0]['samples']
    (osm, osr), (slope, intercept, r) = stats.probplot(
        exp_samples, dist='expon', sparams=(0, 2), fit=True)
    ax_qq.scatter(osm, osr, s=5, alpha=0.5, color='#3498db')
    ax_qq.plot(osm, slope * np.array(osm) + intercept,
               'r-', linewidth=2)
    ax_qq.set_title(f'Q-Q Plot: Exponential\n(R²={r**2:.4f})',
                    fontweight='bold')
    ax_qq.set_xlabel('Theoretical')
    ax_qq.set_ylabel('Sample')
    ax_qq.grid(True, alpha=0.3)
    ax_qq.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab12_inverse_transform.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab12_inverse_transform.png")
    plt.show()
    print("\n✅ Lab 12 Complete!\n")


# ═══════════════════════════════════════════════════════════
# LAB 13
# ═══════════════════════════════════════════════════════════
def lab13_exponential_variates():
    print("\n" + "="*65)
    print("   LAB 13: 5000 EXPONENTIAL VARIATES — INVERSE TRANSFORM")
    print("="*65)

    try:
        n      = int(input("\n  Number of variates  [e.g. 5000]: "))
        lam    = float(input("  Rate parameter λ    [e.g.  1.0]: "))
        seed   = int(input("  LCG Seed            [e.g.   42]: "))
        n_bins = int(input("  Histogram bins      [e.g.   50]: "))
    except ValueError:
        n, lam, seed, n_bins = 5000, 1.0, 42, 50

    mean_ = 1.0 / lam

    # ── Generate U via LCG, then apply inverse transform ──
    U       = lcg_generate(n, seed=seed)
    samples = inv_transform_exponential(U, mean=mean_)

    print(f"\n  Formula used: X = -(1/{lam}) × ln(U)")
    print(f"  = -mean × ln(U)  where mean = {mean_}")

    # ── Statistics ──
    print("\n  ── Statistics ──")
    stat_rows = [
        ["Sample Mean",    f"{np.mean(samples):.6f}",  f"{mean_:.6f}"],
        ["Sample Std Dev", f"{np.std(samples):.6f}",   f"{mean_:.6f}"],
        ["Sample Variance",f"{np.var(samples):.6f}",   f"{mean_**2:.6f}"],
        ["Sample Median",  f"{np.median(samples):.6f}",f"{np.log(2)/lam:.6f}"],
        ["Sample Min",     f"{np.min(samples):.6f}",   "0"],
        ["Sample Max",     f"{np.max(samples):.6f}",   "∞"],
    ]
    print(tabulate(stat_rows,
                   headers=["Metric", "Simulated", "Theoretical"],
                   tablefmt="rounded_grid"))

    # ── KS Test ──
    ks_stat, ks_p = stats.kstest(samples, 'expon', args=(0, mean_))
    print(f"\n  ── KS Test: Exp(λ={lam}) ──")
    print(tabulate([
        ["KS Statistic", f"{ks_stat:.6f}"],
        ["p-value",      f"{ks_p:.6f}"],
        ["Result",       "✅ PASS" if ks_p > 0.05 else "❌ FAIL"],
    ], headers=["Metric", "Value"], tablefmt="rounded_grid"))

    # ── P(X > t) verification ──
    print("\n  ── Survival Probability Verification: P(X > t) = e^{-λt} ──")
    test_t = [0.5, 1.0, 1.5, 2.0, 3.0]
    surv_rows = []
    for t in test_t:
        p_th  = np.exp(-lam * t)
        p_sim = np.mean(samples > t)
        surv_rows.append([t, f"{p_th:.4f}", f"{p_sim:.4f}",
                          f"{abs(p_th - p_sim):.4f}"])
    print(tabulate(surv_rows,
                   headers=["t", "P(X>t) Theory", "P(X>t) Sim", "Error"],
                   tablefmt="rounded_grid"))

    # ── PLOTS ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Lab 13 — Exponential Variates via Inverse Transform (n={n}, λ={lam})",
                 fontsize=13, fontweight='bold', color='#2c3e50')

    x_th = np.linspace(0, np.percentile(samples, 99), 300)
    pdf  = lam * np.exp(-lam * x_th)
    cdf  = 1 - np.exp(-lam * x_th)

    # A: Histogram + PDF
    ax1 = axes[0, 0]
    ax1.hist(samples, bins=n_bins, density=True,
             color='#3498db', alpha=0.75, edgecolor='white',
             label=f'Inverse Transform (n={n})')
    ax1.plot(x_th, pdf, 'r-', linewidth=2.5,
             label=f'Theoretical PDF: λe^{{-λx}}')
    ax1.axvline(np.mean(samples), color='green', linestyle='--',
                linewidth=2, label=f'Sample Mean={np.mean(samples):.3f}')
    ax1.set_title('Histogram + Theoretical PDF', fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # B: ECDF vs CDF
    ax2 = axes[0, 1]
    sorted_s  = np.sort(samples)
    ecdf_vals = np.arange(1, n + 1) / n
    ax2.step(sorted_s, ecdf_vals, color='#3498db', linewidth=1.5,
             label='ECDF (Empirical)')
    ax2.plot(x_th, cdf, 'r-', linewidth=2.5, label='Theoretical CDF')
    ax2.set_title('ECDF vs Theoretical CDF', fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('F(x)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # C: Q-Q Plot
    ax3 = axes[1, 0]
    (osm, osr), (slope, intercept, r_val) = stats.probplot(
        samples, dist='expon', sparams=(0, mean_), fit=True)
    ax3.scatter(osm, osr, s=3, alpha=0.5, color='#3498db')
    ax3.plot(osm, slope * np.array(osm) + intercept,
             'r-', linewidth=2.5, label=f'Fit (R²={r_val**2:.5f})')
    ax3.set_title('Q-Q Plot (Exponential)', fontweight='bold')
    ax3.set_xlabel('Theoretical Quantiles')
    ax3.set_ylabel('Sample Quantiles')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')

    # D: Survival function comparison
    ax4 = axes[1, 1]
    t_vals   = np.linspace(0, np.percentile(samples, 98), 100)
    p_theory = np.exp(-lam * t_vals)
    p_sim    = [np.mean(samples > t) for t in t_vals]
    ax4.plot(t_vals, p_theory, 'r-', linewidth=2.5, label='P(X>t) Theoretical')
    ax4.plot(t_vals, p_sim,    'b--', linewidth=2, label='P(X>t) Simulated')
    ax4.set_title('Survival Function P(X > t)', fontweight='bold')
    ax4.set_xlabel('t')
    ax4.set_ylabel('P(X > t)')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab13_exp_variates.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab13_exp_variates.png")
    plt.show()
    print("\n✅ Lab 13 Complete!\n")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  CIT 324 — Lab 12 & 13: Inverse Transform       ║")
    print("╚══════════════════════════════════════════════════╝")
    print("\n  [12] Lab 12 — Multiple Distributions")
    print("  [13] Lab 13 — Exponential (5000 variates)")
    print("  [0]  Run BOTH")

    try:
        ch = int(input("\n  Choice [0/12/13]: "))
    except ValueError:
        ch = 0

    if ch == 12 or ch == 0:
        lab12_inverse_transform()
    if ch == 13 or ch == 0:
        lab13_exponential_variates()


if __name__ == "__main__":
    main()
