"""
=============================================================
CIT 324 — Lab 11: Linear Congruential Generator (LCG)
=============================================================
Theory (Averill M. Law, Chapter 7 — Random Number Generation):
  LCG Formula: X_{n+1} = (a * X_n + c) mod m
    X_0 = seed (initial value)
    a   = multiplier
    c   = increment
    m   = modulus
  U_n = X_n / m  → uniform random number in [0,1)

  Hull-Dobell Theorem (full period conditions):
    1. c and m are coprime
    2. (a-1) divisible by all prime factors of m
    3. If m divisible by 4, then (a-1) also divisible by 4

  Common good parameters (Law textbook, Table 7.1):
    m = 2^31 - 1 = 2,147,483,647 (Mersenne prime)
    a = 16807  (Park & Miller, 1988)
    c = 0      (pure multiplicative LCG)

  Tests for randomness:
    - Uniformity test (Chi-Square)
    - Independence test (runs test / autocorrelation)
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from tabulate import tabulate


class LCG:
    """Linear Congruential Generator"""

    def __init__(self, seed, a, c, m):
        self.seed = seed
        self.a    = a
        self.c    = c
        self.m    = m
        self.X    = seed

    def next_int(self):
        """Generate next integer X_{n+1} = (a*X_n + c) mod m"""
        self.X = (self.a * self.X + self.c) % self.m
        return self.X

    def next_float(self):
        """Return U_n = X_n / m ∈ [0, 1)"""
        return self.next_int() / self.m

    def generate(self, n):
        """Generate n uniform random numbers"""
        return [self.next_float() for _ in range(n)]

    def generate_with_trace(self, n):
        """Generate n numbers with full trace (X_n, U_n)"""
        trace = []
        for i in range(n):
            xi = self.next_int()
            ui = xi / self.m
            trace.append((i + 1, xi, ui))
        return trace


def chi_square_uniformity_test(numbers, k=10):
    """
    Chi-Square test for uniformity.
    H0: numbers are uniformly distributed on [0,1)
    Divide [0,1) into k equal intervals.
    Expected frequency = n/k per interval.
    """
    n = len(numbers)
    expected = n / k
    counts, _ = np.histogram(numbers, bins=k, range=(0, 1))
    chi2_stat = sum((o - expected)**2 / expected for o in counts)
    df = k - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    critical = stats.chi2.ppf(0.95, df)
    return chi2_stat, p_value, critical, counts, expected


def runs_test(numbers):
    """
    Runs test for independence (above/below median).
    """
    median = np.median(numbers)
    signs  = [1 if x >= median else 0 for x in numbers]
    n1 = sum(signs)
    n2 = len(signs) - n1
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1
    # Expected runs and variance under H0
    mu_r  = (2 * n1 * n2) / (n1 + n2) + 1
    var_r = (2 * n1 * n2 * (2*n1*n2 - n1 - n2)) / ((n1+n2)**2 * (n1+n2-1))
    z     = (runs - mu_r) / np.sqrt(var_r)
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return runs, mu_r, var_r, z, p_val


def period_analysis(seed, a, c, m, max_check=100000):
    """Find the period of the LCG."""
    lcg = LCG(seed, a, c, m)
    seen = {}
    i    = 0
    while i < max_check:
        x = lcg.next_int()
        if x in seen:
            return i - seen[x] + 1
        seen[x] = i
        i += 1
    return max_check  # period > max_check


def lab11_lcg():
    print("\n" + "="*65)
    print("   LAB 11: LINEAR CONGRUENTIAL GENERATOR (LCG)")
    print("="*65)

    # ── Preset parameter sets ──
    presets = {
        "Park & Miller (1988) — Recommended": (16807,       0, 2**31 - 1),
        "Numerical Recipes":                  (1664525,  1013904223, 2**32),
        "RANDU (Bad — avoid!)":               (65539,       0, 2**31),
        "Small demo (full period)":           (5,           3, 16),
    }

    print("\n  Available Parameter Presets:")
    for i, (name, (a, c, m)) in enumerate(presets.items(), 1):
        print(f"  [{i}] {name}  (a={a}, c={c}, m={m})")
    print("  [0] Enter custom parameters")

    try:
        choice = int(input("\n  Select preset [0-4]: "))
    except ValueError:
        choice = 1

    if choice == 0:
        try:
            seed_ = int(input("  Seed (X_0)    : "))
            a_    = int(input("  Multiplier (a): "))
            c_    = int(input("  Increment  (c): "))
            m_    = int(input("  Modulus    (m): "))
        except ValueError:
            seed_, a_, c_, m_ = 1, 16807, 0, 2**31 - 1
    else:
        preset_list = list(presets.values())
        a_, c_, m_  = preset_list[min(choice, 4) - 1]
        seed_       = int(input(f"  Enter seed (X_0) [e.g. 1]: ") or 1)

    try:
        n_ = int(input("  How many numbers to generate? [e.g. 10000]: "))
    except ValueError:
        n_ = 10000

    # ── Generate ──
    lcg = LCG(seed_, a_, c_, m_)

    print(f"\n  ── LCG Parameters ──")
    print(f"  Formula: X_{{n+1}} = ({a_} × X_n + {c_}) mod {m_}")
    print(f"  Seed: {seed_},   N: {n_}")

    # Show first 20 values as trace
    lcg_trace = LCG(seed_, a_, c_, m_)
    trace = lcg_trace.generate_with_trace(20)
    print("\n  ── First 20 Generated Values ──")
    print(tabulate(trace,
                   headers=["n", "X_n (integer)", "U_n = X_n/m (uniform)"],
                   tablefmt="rounded_grid",
                   floatfmt=".8f"))

    # Generate full set
    numbers = lcg.generate(n_)

    # ── Period ──
    period = period_analysis(seed_, a_, c_, m_, max_check=min(m_, 200000))
    print(f"\n  ── Period Analysis ──")
    print(f"  Estimated Period : {period:,}")
    print(f"  Modulus m        : {m_:,}")
    print(f"  Period / m       : {period/m_:.6f}")

    # ── Chi-Square Test ──
    chi2, p_chi, crit, counts, expected = chi_square_uniformity_test(numbers, k=10)
    print(f"\n  ── Chi-Square Uniformity Test (k=10 bins) ──")
    chi_rows = [(f"[{i/10:.1f}, {(i+1)/10:.1f})", counts[i], f"{expected:.1f}")
                for i in range(10)]
    print(tabulate(chi_rows, headers=["Interval", "Observed", "Expected"],
                   tablefmt="rounded_grid"))
    print(f"\n  χ² statistic : {chi2:.4f}")
    print(f"  Critical val  : {crit:.4f}  (α=0.05, df=9)")
    print(f"  p-value       : {p_chi:.4f}")
    if p_chi > 0.05:
        print("  ✅ PASS: Cannot reject H0 (appears uniform)")
    else:
        print("  ❌ FAIL: Reject H0 (not uniform)")

    # ── Runs Test ──
    runs, mu_r, var_r, z, p_runs = runs_test(numbers)
    print(f"\n  ── Runs Test for Independence ──")
    runs_rows = [
        ["Observed Runs",    f"{runs}"],
        ["Expected Runs",    f"{mu_r:.2f}"],
        ["Variance",         f"{var_r:.2f}"],
        ["Z-statistic",      f"{z:.4f}"],
        ["p-value",          f"{p_runs:.4f}"],
        ["Result",           "✅ PASS" if p_runs > 0.05 else "❌ FAIL"],
    ]
    print(tabulate(runs_rows, headers=["Metric", "Value"], tablefmt="rounded_grid"))

    # ── Statistics ──
    print(f"\n  ── Descriptive Statistics (should match Uniform[0,1]) ──")
    stat_rows = [
        ["Mean",     f"{np.mean(numbers):.6f}",  "0.500000"],
        ["Variance", f"{np.var(numbers):.6f}",   "0.083333"],
        ["Min",      f"{np.min(numbers):.6f}",   "0.000000"],
        ["Max",      f"{np.max(numbers):.6f}",   "1.000000"],
    ]
    print(tabulate(stat_rows, headers=["Metric", "Simulated", "Theoretical"],
                   tablefmt="rounded_grid"))

    # ── PLOTS ──
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"Lab 11 — LCG: a={a_}, c={c_}, m={m_}",
                 fontsize=14, fontweight='bold', color='#2c3e50')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # A: Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(numbers, bins=20, color='#3498db', alpha=0.8, edgecolor='white',
             density=True)
    ax1.axhline(1.0, color='red', linewidth=2, linestyle='--', label='Expected = 1.0')
    ax1.set_title('Histogram of U_n\n(Should be flat)', fontweight='bold')
    ax1.set_xlabel('U_n')
    ax1.set_ylabel('Density')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # B: Scatter (U_n vs U_{n+1}) — independence check
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(numbers[:-1], numbers[1:], s=1, alpha=0.3, color='#8e44ad')
    ax2.set_title('U_n vs U_{n+1}\n(Should be scattered)', fontweight='bold')
    ax2.set_xlabel('U_n')
    ax2.set_ylabel('U_{n+1}')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # C: 3D scatter (U_n, U_{n+1}, U_{n+2})
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    n_3d = min(2000, len(numbers) - 2)
    ax3.scatter(numbers[:n_3d], numbers[1:n_3d+1], numbers[2:n_3d+2],
                s=1, alpha=0.3, c='#e74c3c')
    ax3.set_title('3D Scatter\n(U_n, U_{n+1}, U_{n+2})', fontweight='bold')
    ax3.set_xlabel('U_n')
    ax3.set_ylabel('U_{n+1}')
    ax3.set_zlabel('U_{n+2}')

    # D: Autocorrelation
    ax4 = fig.add_subplot(gs[1, 0])
    lags  = range(1, 21)
    acorr = [np.corrcoef(numbers[:-k], numbers[k:])[0, 1] for k in lags]
    ax4.bar(lags, acorr, color='#2ecc71', alpha=0.85, edgecolor='white')
    ax4.axhline(0, color='black', linewidth=1)
    ax4.axhline(1.96/np.sqrt(n_), color='red', linestyle='--', label='95% CI')
    ax4.axhline(-1.96/np.sqrt(n_), color='red', linestyle='--')
    ax4.set_title('Autocorrelation (should be ~0)', fontweight='bold')
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('Correlation')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')

    # E: Running mean convergence
    ax5 = fig.add_subplot(gs[1, 1])
    run_mean = np.cumsum(numbers) / np.arange(1, n_+1)
    ax5.plot(run_mean, color='#3498db', linewidth=1.2)
    ax5.axhline(0.5, color='red', linestyle='--', linewidth=2, label='True mean = 0.5')
    ax5.set_title('Running Mean → 0.5', fontweight='bold')
    ax5.set_xlabel('n')
    ax5.set_ylabel('Running Mean')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_facecolor('#f8f9fa')

    # F: Chi-Square bins
    ax6 = fig.add_subplot(gs[1, 2])
    bin_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
    bars = ax6.bar(range(10), counts, color='#f39c12', alpha=0.85,
                   edgecolor='white', label='Observed')
    ax6.axhline(expected, color='red', linestyle='--',
                linewidth=2, label=f'Expected={expected:.0f}')
    ax6.set_title(f'Chi-Square Uniformity\nχ²={chi2:.2f}, p={p_chi:.3f}',
                  fontweight='bold')
    ax6.set_xticks(range(10))
    ax6.set_xticklabels(bin_labels, rotation=45, fontsize=7)
    ax6.set_ylabel('Count')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_facecolor('#f8f9fa')

    plt.savefig('/mnt/user-data/outputs/lab11_lcg.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab11_lcg.png")
    plt.show()
    print("\n✅ Lab 11 Complete!\n")


if __name__ == "__main__":
    lab11_lcg()
