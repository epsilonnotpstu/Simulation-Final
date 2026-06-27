"""
=============================================================
CIT 324 - Simulation and Modeling Sessional
Lab 09: Empirical Input Modeling — MLE + Q-Q Plot
Lab 10: Machine Breakdown & Maintenance Simulation
=============================================================
Theory (Averill M. Law, Chapter 6 — Input Modeling):
  Lab 09:
  - Fit raw data to candidate distributions using MLE
    (scipy's fit() uses Maximum Likelihood Estimation)
  - Validate with Q-Q plots and goodness-of-fit tests
  - Distributions compared: Normal, Weibull, Lognormal, Exponential

  Lab 10:
  - 5 machines; breakdown ~ Exponential (memoryless)
  - Repair time ~ Weibull (shape parameter models wear)
  - Track: downtime per machine, utilization, throughput
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from tabulate import tabulate


# ═══════════════════════════════════════════════════════════
# LAB 09 — EMPIRICAL INPUT MODELING
# ═══════════════════════════════════════════════════════════

def lab09_input_modeling(seed=42):
    """
    Simulate 'raw' call-center arrival data, then fit distributions.
    In a real lab, this data would come from observations.
    """
    rng = np.random.default_rng(seed)

    # ── Generate raw data (simulate call center inter-arrival times) ──
    # True underlying: Lognormal(μ=1.5, σ=0.6) in minutes
    true_mu, true_sigma = 1.5, 0.6
    n_obs = 200
    raw_data = rng.lognormal(true_mu, true_sigma, n_obs)

    print("\n" + "="*62)
    print("   LAB 09: EMPIRICAL INPUT MODELING")
    print("="*62)
    print(f"\n   Dataset: {n_obs} inter-arrival times (minutes)")
    print(f"   (True underlying: Lognormal, μ={true_mu}, σ={true_sigma})")

    # ── Descriptive Statistics ──
    print("\n  ── Descriptive Statistics ──")
    desc_rows = [
        ["Count",       f"{n_obs}"],
        ["Min",         f"{raw_data.min():.4f}"],
        ["Max",         f"{raw_data.max():.4f}"],
        ["Mean",        f"{raw_data.mean():.4f}"],
        ["Median",      f"{np.median(raw_data):.4f}"],
        ["Std Dev",     f"{raw_data.std():.4f}"],
        ["Variance",    f"{raw_data.var():.4f}"],
        ["Skewness",    f"{stats.skew(raw_data):.4f}"],
        ["Kurtosis",    f"{stats.kurtosis(raw_data):.4f}"],
    ]
    print(tabulate(desc_rows, headers=["Statistic", "Value"],
                   tablefmt="rounded_grid"))

    # ── Fit candidate distributions using MLE ──
    candidates = {
        'Normal'      : stats.norm,
        'Lognormal'   : stats.lognorm,
        'Weibull'     : stats.weibull_min,
        'Exponential' : stats.expon,
    }

    fit_results = {}
    print("\n  ── MLE Parameter Fitting ──")
    fit_rows = []
    for name, dist in candidates.items():
        params = dist.fit(raw_data, floc=0) if name == 'Exponential' else dist.fit(raw_data)
        # KS test
        ks_stat, ks_p = stats.kstest(raw_data, dist.cdf, args=params)
        # AIC = 2k - 2*log(L)  (lower = better)
        log_l = np.sum(dist.logpdf(raw_data, *params))
        k     = len(params)
        aic   = 2*k - 2*log_l
        fit_results[name] = {
            'params'  : params,
            'ks_stat' : ks_stat,
            'ks_p'    : ks_p,
            'aic'     : aic,
            'dist'    : dist,
        }
        fit_rows.append([name,
                         ', '.join(f"{p:.4f}" for p in params),
                         f"{ks_stat:.4f}",
                         f"{ks_p:.4f}",
                         f"{aic:.2f}"])

    print(tabulate(fit_rows,
                   headers=["Distribution", "MLE Params",
                             "KS Stat", "KS p-value", "AIC"],
                   tablefmt="rounded_grid"))

    # Identify best fit
    best = min(fit_results, key=lambda k: fit_results[k]['aic'])
    print(f"\n  ✅ Best fit by AIC: {best}")
    print(f"     (Lower AIC = better fit; KS p > 0.05 = not rejected)")

    # ── PLOT ──
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Lab 09 — Empirical Input Modeling (MLE + Validation)",
                 fontsize=14, fontweight='bold', color='#2c3e50')

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    # ─ Histogram with fitted PDFs ─
    ax_hist = fig.add_subplot(gs[0, 0:3])
    ax_hist.hist(raw_data, bins=30, density=True,
                 color='#bdc3c7', edgecolor='white',
                 label='Observed Data', alpha=0.9)
    x_fit = np.linspace(raw_data.min(), raw_data.max(), 300)
    clrs  = ['#3498db','#e74c3c','#2ecc71','#f39c12']
    for (name, res), c in zip(fit_results.items(), clrs):
        pdf = res['dist'].pdf(x_fit, *res['params'])
        ax_hist.plot(x_fit, pdf, linewidth=2.5, color=c, label=name)
    ax_hist.set_title('Raw Data Histogram + MLE Fitted PDFs', fontweight='bold')
    ax_hist.set_xlabel('Inter-Arrival Time (min)')
    ax_hist.set_ylabel('Density')
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.set_facecolor('#f8f9fa')

    # ─ AIC bar chart ─
    ax_aic = fig.add_subplot(gs[0, 3])
    names = list(fit_results.keys())
    aics  = [fit_results[n]['aic'] for n in names]
    bar_colors = ['#27ae60' if n == best else '#95a5a6' for n in names]
    ax_aic.barh(names, aics, color=bar_colors, edgecolor='white')
    ax_aic.set_title('AIC Comparison\n(Lower = Better)', fontweight='bold')
    ax_aic.set_xlabel('AIC')
    ax_aic.grid(True, alpha=0.3, axis='x')
    ax_aic.set_facecolor('#f8f9fa')

    # ─ Q-Q Plots for each distribution ─
    for i, (name, res) in enumerate(fit_results.items()):
        ax = fig.add_subplot(gs[1, i])
        osm, osr = stats.probplot(raw_data, dist=res['dist'],
                                  sparams=res['params'][:-2]
                                  if name != 'Normal' else res['params'],
                                  fit=True)[0]
        _, (slope, intercept, r) = stats.probplot(
            raw_data, dist=res['dist'],
            sparams=res['params'],
            fit=True)
        ax.scatter(osm, osr, color='#3498db', s=12, alpha=0.7)
        ax.plot(osm, np.array(osm)*slope + intercept,
                'r-', linewidth=2)
        clr = '#27ae60' if name == best else '#2c3e50'
        ax.set_title(f'{name}\nKS p={res["ks_p"]:.3f}',
                     fontweight='bold', color=clr)
        ax.set_xlabel('Theoretical')
        ax.set_ylabel('Sample')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')

    plt.savefig('/mnt/user-data/outputs/lab09_input_modeling.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab09_input_modeling.png")
    plt.show()


# ═══════════════════════════════════════════════════════════
# LAB 10 — MACHINE BREAKDOWN & MAINTENANCE
# ═══════════════════════════════════════════════════════════

def lab10_machine_breakdown(
    n_machines         = 5,
    mean_breakdown     = 50.0,   # hours (Exponential)
    weibull_shape      = 2.0,    # k — shape for repair time
    weibull_scale      = 8.0,    # λ — scale for repair time (hours)
    sim_hours          = 2000,   # total simulation time
    seed               = 42
):
    """
    Workshop with n_machines identical machines.
    Breakdown inter-event time: Exponential(mean_breakdown)
    Repair time: Weibull(shape, scale)
    One repair team (single server).
    """
    rng = np.random.default_rng(seed)

    print("\n" + "="*62)
    print("   LAB 10: MACHINE BREAKDOWN & MAINTENANCE SIMULATION")
    print("="*62)
    print(f"\n   Number of Machines   : {n_machines}")
    print(f"   Breakdown Time       : Exponential(μ={mean_breakdown} hr)")
    print(f"   Repair Time          : Weibull(k={weibull_shape}, λ={weibull_scale})")
    print(f"   Simulation Duration  : {sim_hours} hours")

    # ── Per-machine tracking ──
    machine_state   = ['running'] * n_machines   # 'running' or 'down'
    next_breakdown  = [rng.exponential(mean_breakdown) for _ in range(n_machines)]
    repair_end      = [float('inf')] * n_machines

    total_downtime  = [0.0] * n_machines
    breakdown_count = [0]   * n_machines
    repair_queue    = []    # machines waiting to be repaired
    repair_busy     = False
    current_machine_in_repair = None
    repair_start_time         = None

    sim_clock        = 0.0
    all_events       = []   # for timeline plot

    while sim_clock < sim_hours:

        # ── Collect all candidate event times ──
        candidates = []
        for i in range(n_machines):
            if machine_state[i] == 'running':
                candidates.append((next_breakdown[i], 'breakdown', i))
        # Repair completion
        for i in range(n_machines):
            if repair_end[i] < float('inf'):
                candidates.append((repair_end[i], 'repair_done', i))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        t_next, event_type, m_idx = candidates[0]

        if t_next > sim_hours:
            sim_clock = sim_hours
            break

        # ── Accumulate downtime up to t_next ──
        for i in range(n_machines):
            if machine_state[i] == 'down':
                total_downtime[i] += (t_next - sim_clock)

        sim_clock = t_next

        if event_type == 'breakdown':
            machine_state[m_idx]  = 'down'
            breakdown_count[m_idx] += 1
            next_breakdown[m_idx]  = float('inf')  # no new breakdown until repaired

            if not repair_busy:
                # Repair starts immediately
                repair_busy               = True
                current_machine_in_repair = m_idx
                repair_start_time         = sim_clock
                repair_t = rng.weibull(weibull_shape) * weibull_scale
                repair_end[m_idx] = sim_clock + repair_t
            else:
                repair_queue.append(m_idx)

            all_events.append((sim_clock, m_idx, 'BD'))

        elif event_type == 'repair_done':
            machine_state[m_idx] = 'running'
            repair_end[m_idx]    = float('inf')
            next_breakdown[m_idx] = sim_clock + rng.exponential(mean_breakdown)
            repair_busy           = False
            all_events.append((sim_clock, m_idx, 'RP'))

            # Start repairing next in queue
            if repair_queue:
                next_m    = repair_queue.pop(0)
                repair_busy               = True
                current_machine_in_repair = next_m
                repair_t = rng.weibull(weibull_shape) * weibull_scale
                repair_end[next_m] = sim_clock + repair_t

    # ── Final downtime for any still-down machines ──
    for i in range(n_machines):
        if machine_state[i] == 'down':
            total_downtime[i] += (sim_hours - sim_clock)

    # ── Performance Metrics ──
    availability = [(sim_hours - dt) / sim_hours for dt in total_downtime]
    avg_avail    = np.mean(availability)
    total_dt     = sum(total_downtime)
    throughput   = avg_avail * n_machines   # effective machines running on avg

    print("\n  ── Per-Machine Results ──")
    machine_rows = [
        [f"Machine {i+1}",
         breakdown_count[i],
         f"{total_downtime[i]:.2f} hr",
         f"{(sim_hours-total_downtime[i]):.2f} hr",
         f"{availability[i]*100:.1f}%"]
        for i in range(n_machines)
    ]
    print(tabulate(machine_rows,
                   headers=["Machine", "Breakdowns", "Downtime",
                             "Uptime", "Availability"],
                   tablefmt="rounded_grid"))

    print("\n  ── System Summary ──")
    summary_rows = [
        ["Total Downtime (all machines)",  f"{total_dt:.2f} hr"],
        ["Average Machine Availability",   f"{avg_avail*100:.2f}%"],
        ["Effective Throughput",           f"{throughput:.2f} machines"],
        ["Total Breakdowns",               f"{sum(breakdown_count)}"],
        ["Mean Repairs (per machine)",     f"{np.mean(breakdown_count):.2f}"],
    ]
    print(tabulate(summary_rows, headers=["Metric", "Value"],
                   tablefmt="rounded_grid"))

    # ── PLOT ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Lab 10 — Machine Breakdown & Maintenance Simulation",
                 fontsize=14, fontweight='bold', color='#2c3e50')

    machines   = [f"M{i+1}" for i in range(n_machines)]
    colors     = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']

    # ─ A: Downtime per machine ─
    ax1 = axes[0, 0]
    bars = ax1.bar(machines, total_downtime, color=colors, edgecolor='white',
                   alpha=0.85)
    ax1.bar_label(bars, fmt='%.1f hr', fontsize=9)
    ax1.set_title('Total Downtime per Machine', fontweight='bold')
    ax1.set_ylabel('Downtime (hours)')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('#f8f9fa')

    # ─ B: Availability ─
    ax2 = axes[0, 1]
    avail_pct = [a*100 for a in availability]
    bars2 = ax2.bar(machines, avail_pct, color=colors, edgecolor='white', alpha=0.85)
    ax2.bar_label(bars2, fmt='%.1f%%', fontsize=9)
    ax2.axhline(avg_avail*100, color='red', linestyle='--',
                linewidth=2, label=f'Avg={avg_avail*100:.1f}%')
    ax2.set_ylim(0, 110)
    ax2.set_title('Machine Availability', fontweight='bold')
    ax2.set_ylabel('Availability (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#f8f9fa')

    # ─ C: Breakdown count ─
    ax3 = axes[1, 0]
    bars3 = ax3.bar(machines, breakdown_count, color=colors,
                    edgecolor='white', alpha=0.85)
    ax3.bar_label(bars3, fontsize=9)
    ax3.set_title('Number of Breakdowns per Machine', fontweight='bold')
    ax3.set_ylabel('Breakdowns')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_facecolor('#f8f9fa')

    # ─ D: Uptime vs Downtime (stacked) ─
    ax4 = axes[1, 1]
    uptime = [sim_hours - dt for dt in total_downtime]
    ax4.bar(machines, uptime, color='#27ae60', label='Uptime', alpha=0.85)
    ax4.bar(machines, total_downtime, bottom=uptime,
            color='#e74c3c', label='Downtime', alpha=0.85)
    ax4.set_title('Uptime vs Downtime (Stacked)', fontweight='bold')
    ax4.set_ylabel('Hours')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab10_machine_breakdown.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab10_machine_breakdown.png")
    plt.show()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  CIT 324 — Lab 09 & 10: Modeling & Machines     ║")
    print("╚══════════════════════════════════════════════════╝")
    print("\n  [9]  Lab 09 — Empirical Input Modeling")
    print("  [10] Lab 10 — Machine Breakdown")
    print("  [0]  Run BOTH")

    try:
        ch = int(input("\n  Choice [0/9/10]: "))
    except ValueError:
        ch = 0

    if ch == 9 or ch == 0:
        lab09_input_modeling()

    if ch == 10 or ch == 0:
        try:
            nm   = int(input("\n  Lab10 — Number of machines    [e.g.   5]: "))
            mbd  = float(input("  Lab10 — Mean breakdown (hours)[e.g.  50]: "))
            wk   = float(input("  Lab10 — Weibull shape k       [e.g. 2.0]: "))
            wl   = float(input("  Lab10 — Weibull scale λ (hrs) [e.g. 8.0]: "))
            sh   = int(input("  Lab10 — Simulation hours      [e.g.2000]: "))
        except ValueError:
            nm, mbd, wk, wl, sh = 5, 50.0, 2.0, 8.0, 2000
        lab10_machine_breakdown(
            n_machines=nm, mean_breakdown=mbd,
            weibull_shape=wk, weibull_scale=wl,
            sim_hours=sh
        )

    print("\n✅ Labs 09-10 Complete!\n")


if __name__ == "__main__":
    main()
