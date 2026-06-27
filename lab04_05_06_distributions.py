"""
=============================================================
CIT 324 - Simulation and Modeling Sessional
Lab 04: Bernoulli Distribution — Microchip Quality Control
Lab 05: Binomial Distribution  — Network Packet Delivery
Lab 06: Poisson Distribution   — Customer Care Call Center
=============================================================
Theory (Averill M. Law, Chapter 8):
  - Bernoulli: Binary outcome (success/fail), P(X=1)=p
  - Binomial:  n independent Bernoulli trials, X ~ B(n,p)
  - Poisson:   Events in fixed interval, X ~ Poisson(λ)
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from tabulate import tabulate
from math import exp, factorial, comb


# ═══════════════════════════════════════════════════════════
# LAB 04 — BERNOULLI DISTRIBUTION
# ═══════════════════════════════════════════════════════════

def lab04_bernoulli(p=0.85, n_chips=1000, seed=42):
    """
    Microchip quality control.
    X = 1 (pass), X = 0 (fail)
    P(X=1) = p,  P(X=0) = 1-p
    E[X] = p,    Var(X) = p(1-p)
    """
    rng     = np.random.default_rng(seed)
    results = rng.binomial(1, p, n_chips)  # simulate n_chips tests

    print("\n" + "="*60)
    print("   LAB 04: BERNOULLI — Microchip Quality Control")
    print("="*60)
    print(f"\n   Probability of passing (p) : {p}")
    print(f"   Chips tested               : {n_chips}")

    # ── Theoretical values ──
    P_fail  = 1 - p          # P(X=0)
    E_X     = p              # E[X]
    Var_X   = p * (1 - p)   # Var(X)

    # ── Simulated values ──
    sim_pass = np.sum(results == 1)
    sim_fail = np.sum(results == 0)
    sim_p    = sim_pass / n_chips

    print("\n  ── Probability Analysis ──")
    rows_prob = [
        ["P(X=1) — Chip Passes", f"{p:.4f}",       f"{sim_pass/n_chips:.4f}"],
        ["P(X=0) — Chip Fails",  f"{P_fail:.4f}",  f"{sim_fail/n_chips:.4f}"],
    ]
    print(tabulate(rows_prob,
                   headers=["Outcome", "Theoretical", "Simulated"],
                   tablefmt="rounded_grid"))

    print("\n  ── Statistical Metrics ──")
    rows_stat = [
        ["E[X]   (Expected Value)", f"{E_X:.4f}", f"{np.mean(results):.4f}"],
        ["Var(X) (Variance)",       f"{Var_X:.4f}", f"{np.var(results):.4f}"],
        ["Std(X)",                  f"{np.sqrt(Var_X):.4f}", f"{np.std(results):.4f}"],
    ]
    print(tabulate(rows_stat,
                   headers=["Metric", "Theoretical", "Simulated"],
                   tablefmt="rounded_grid"))

    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Lab 04 — Bernoulli Distribution: Microchip QC",
                 fontsize=13, fontweight='bold', color='#2c3e50')

    # PMF bar chart
    ax1 = axes[0]
    xv      = [0, 1]
    theory  = [P_fail, p]
    sim_pmf = [sim_fail/n_chips, sim_pass/n_chips]
    x_pos   = np.arange(2)
    bw      = 0.35
    ax1.bar(x_pos - bw/2, theory,  bw, label='Theoretical', color='#3498db', alpha=0.85)
    ax1.bar(x_pos + bw/2, sim_pmf, bw, label='Simulated',   color='#e74c3c', alpha=0.85)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['X=0 (Fail)', 'X=1 (Pass)'])
    ax1.set_ylabel('Probability')
    ax1.set_title('Bernoulli PMF (p = 0.85)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('#f8f9fa')

    # Convergence: running proportion of passes
    ax2 = axes[1]
    run_avg = np.cumsum(results) / np.arange(1, n_chips + 1)
    ax2.plot(run_avg, color='#2ecc71', linewidth=1.5, label='Running P(pass)')
    ax2.axhline(p, color='#e74c3c', linestyle='--', linewidth=2, label=f'True p={p}')
    ax2.set_xlabel('Number of Chips Tested')
    ax2.set_ylabel('Proportion Passing')
    ax2.set_title('Convergence to True p (Law of Large Numbers)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab04_bernoulli.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab04_bernoulli.png")
    plt.show()


# ═══════════════════════════════════════════════════════════
# LAB 05 — BINOMIAL DISTRIBUTION
# ═══════════════════════════════════════════════════════════

def lab05_binomial(n=15, p=0.90, example_data=None, seed=42):
    """
    Network packet delivery.
    X ~ Binomial(n=15, p=0.90)
    """
    if example_data is None:
        example_data = [14, 13, 15, 12, 14]

    rng = np.random.default_rng(seed)

    print("\n" + "="*60)
    print("   LAB 05: BINOMIAL — Network Packet Delivery")
    print("="*60)
    print(f"\n   n (packets per batch) : {n}")
    print(f"   p (success prob)      : {p}")
    print(f"   q (failure prob)      : {1-p:.2f}")
    print(f"   Example batches data  : {example_data}")

    # ── Parameters ──
    q    = 1 - p
    mu   = n * p
    var  = n * p * q
    std  = np.sqrt(var)

    # ── P(X = 13) ──
    P_13 = comb(n, 13) * (p**13) * (q**(n-13))

    # ── Example data stats ──
    data_mean = np.mean(example_data)

    print("\n  ── Parameter Identification ──")
    print(tabulate([["n", n], ["p", p], ["q", q]],
                   headers=["Parameter", "Value"],
                   tablefmt="rounded_grid"))

    print("\n  ── Theoretical Metrics ──")
    print(tabulate([
        ["E[X] = np",        f"{mu:.4f}"],
        ["Var(X) = npq",     f"{var:.4f}"],
        ["Std(X) = √(npq)",  f"{std:.4f}"],
        ["P(X=13)",          f"{P_13:.6f}"],
    ], headers=["Metric", "Value"], tablefmt="rounded_grid"))

    print("\n  ── Data Verification ──")
    print(tabulate([
        ["Sample Mean (data)",    f"{data_mean:.2f}"],
        ["Theoretical Mean (np)", f"{mu:.2f}"],
        ["Difference",            f"{abs(data_mean - mu):.2f}"],
    ], headers=["Metric", "Value"], tablefmt="rounded_grid"))

    # ── Full PMF ──
    k_vals = np.arange(0, n + 1)
    pmf    = [comb(n, k) * (p**k) * (q**(n-k)) for k in k_vals]

    # ── Simulate many batches ──
    sims = rng.binomial(n, p, 5000)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Lab 05 — Binomial Distribution: Network Packet Delivery",
                 fontsize=13, fontweight='bold', color='#2c3e50')

    # PMF
    ax1 = axes[0]
    ax1.bar(k_vals, pmf, color='#3498db', alpha=0.85, label='Theoretical PMF')
    ax1.axvline(mu, color='#e74c3c', linestyle='--',
                linewidth=2, label=f'μ = np = {mu}')
    ax1.scatter([13], [P_13], color='#f39c12', s=120, zorder=5,
                label=f'P(X=13) = {P_13:.4f}')
    ax1.set_xlabel('Number of Packets Delivered (k)')
    ax1.set_ylabel('Probability P(X=k)')
    ax1.set_title(f'Binomial PMF — B({n}, {p})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('#f8f9fa')

    # Simulation histogram vs theory
    ax2 = axes[1]
    sim_counts, _ = np.histogram(sims, bins=np.arange(-0.5, n+1.5))
    sim_pmf = sim_counts / len(sims)
    ax2.bar(k_vals, pmf,     alpha=0.6, color='#3498db', label='Theoretical')
    ax2.bar(k_vals, sim_pmf, alpha=0.6, color='#e74c3c', label='Simulated (5000 batches)')
    ax2.scatter(example_data,
                [0.01]*len(example_data),
                color='#2ecc71', s=100, zorder=5,
                marker='D', label='Example data points')
    ax2.set_xlabel('Packets Delivered')
    ax2.set_ylabel('Probability')
    ax2.set_title('Simulated vs Theoretical PMF')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab05_binomial.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab05_binomial.png")
    plt.show()


# ═══════════════════════════════════════════════════════════
# LAB 06 — POISSON DISTRIBUTION
# ═══════════════════════════════════════════════════════════

def poisson_pmf(k, lam):
    """P(X=k) = e^(-λ) * λ^k / k!"""
    return (exp(-lam) * (lam ** k)) / factorial(k)


def lab06_poisson(lambda_vals=None, max_k=15, seed=42):
    """
    Customer care center receives calls.
    Default λ = 5 calls/hour.
    Show PMF for λ = 5, 10, 15.
    """
    if lambda_vals is None:
        lambda_vals = [5, 10, 15]

    rng = np.random.default_rng(seed)

    print("\n" + "="*60)
    print("   LAB 06: POISSON — Customer Care Call Center")
    print("="*60)

    # ── Table for λ=5, k=0 to 10 ──
    lam5  = lambda_vals[0]
    k_arr = list(range(max_k + 1))
    pmf5  = [poisson_pmf(k, lam5) for k in k_arr]

    print(f"\n  Probability Table for λ = {lam5} calls/hour")
    rows = [(k, f"{p:.6f}", f"{'▓' * int(p*100)}") for k, p in zip(k_arr, pmf5)]
    print(tabulate(rows,
                   headers=["Calls (k)", "P(X=k)", "Visual"],
                   tablefmt="rounded_grid"))

    print(f"\n  ── Metrics for λ = {lam5} ──")
    print(tabulate([
        ["E[X] = λ",     f"{lam5}"],
        ["Var(X) = λ",   f"{lam5}"],
        ["Std(X) = √λ",  f"{np.sqrt(lam5):.4f}"],
    ], headers=["Metric", "Value"], tablefmt="rounded_grid"))

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lab 06 — Poisson Distribution: Customer Care Center",
                 fontsize=13, fontweight='bold', color='#2c3e50')

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    k_plot = np.arange(0, max_k + 1)

    for ax, lam, clr in zip(axes, lambda_vals, colors):
        pmf = [poisson_pmf(k, lam) for k in k_plot]
        # Simulate
        sim = rng.poisson(lam, 5000)
        sim_pmf, _ = np.histogram(sim, bins=np.arange(-0.5, max_k+1.5),
                                  density=True)
        sim_pmf = sim_pmf * (sim_pmf.sum() * (1/sim_pmf.sum()))

        ax.bar(k_plot, pmf, color=clr, alpha=0.8,
               label=f'Theoretical PMF')
        ax.plot(k_plot, pmf, 'o-', color='#2c3e50', linewidth=1.5,
                markersize=5, label='PMF curve')
        ax.axvline(lam, color='black', linestyle='--',
                   linewidth=2, label=f'μ = λ = {lam}')
        ax.set_title(f'Poisson(λ={lam})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Number of Calls (k)')
        ax.set_ylabel('P(X=k)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab06_poisson.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab06_poisson.png")
    plt.show()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  CIT 324 — Lab 04, 05, 06: Distributions        ║")
    print("╚══════════════════════════════════════════════════╝")

    print("\n  Select Lab:")
    print("  [4] Lab 04 — Bernoulli (Microchip QC)")
    print("  [5] Lab 05 — Binomial  (Network Packets)")
    print("  [6] Lab 06 — Poisson   (Call Center)")
    print("  [0] Run ALL")

    try:
        choice = int(input("\n  Enter choice [0/4/5/6]: "))
    except ValueError:
        choice = 0

    if choice == 4 or choice == 0:
        try:
            p4 = float(input("\n  Lab04 — P(pass) for microchip [e.g. 0.85]: "))
            n4 = int(input("  Lab04 — Number of chips to simulate [e.g. 1000]: "))
        except ValueError:
            p4, n4 = 0.85, 1000
        lab04_bernoulli(p=p4, n_chips=n4)

    if choice == 5 or choice == 0:
        try:
            n5 = int(input("\n  Lab05 — Packets per batch n [e.g. 15]: "))
            p5 = float(input("  Lab05 — Success prob p [e.g. 0.90]: "))
        except ValueError:
            n5, p5 = 15, 0.90
        lab05_binomial(n=n5, p=p5)

    if choice == 6 or choice == 0:
        try:
            l1 = float(input("\n  Lab06 — λ₁ (primary rate) [e.g. 5]: "))
            l2 = float(input("  Lab06 — λ₂ (second rate)  [e.g. 10]: "))
            l3 = float(input("  Lab06 — λ₃ (third rate)   [e.g. 15]: "))
        except ValueError:
            l1, l2, l3 = 5, 10, 15
        lab06_poisson(lambda_vals=[l1, l2, l3])

    print("\n✅ Labs 04–06 Complete!\n")


if __name__ == "__main__":
    main()
