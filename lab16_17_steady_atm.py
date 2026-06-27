"""
=============================================================
CIT 324 — Lab 16: Steady-State Analysis (Warm-Up Detection)
         Lab 17: Bank ATM System Simulation
=============================================================
Theory (Averill M. Law, Chapters 9 & 2):

  Lab 16 — Steady-State Analysis:
    Non-terminating simulations must detect warm-up period.
    Welch's Method: moving average of replication means.
    Idea: plot running average of queue length across many
    short replications; when it "flattens", that's steady-state.
    Delete observations before the inflection point.

  Lab 17 — Bank ATM:
    Multi-server queue: M/M/c
    c servers (ATMs), FIFO discipline
    Performance: P(wait), expected wait, utilization per ATM
    M/M/c formulas:
      ρ = λ/(c×μ)
      P0: computed via Erlang-C formula
      Wq = C(c,λ/μ) / (c×μ - λ)  where C = Erlang-C probability
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate
import math


# ═══════════════════════════════════════════════════════════
# LAB 16 — STEADY-STATE ANALYSIS
# ═══════════════════════════════════════════════════════════

def run_queue_sim_stream(mia, ms, duration, seed):
    """Run M/M/1, collect (time, queue_length) stream."""
    rng         = np.random.default_rng(seed)
    clock       = 0.0
    server_busy = False
    queue       = []
    arrival_t   = -mia * np.log(rng.random())
    depart_t    = float('inf')
    snapshots   = []  # (time, num_in_system)

    while clock < duration:
        if arrival_t <= depart_t and arrival_t <= duration:
            clock = arrival_t
            if not server_busy:
                server_busy = True
                depart_t = clock + (-ms * np.log(rng.random()))
            else:
                queue.append(clock)
            snapshots.append((clock, len(queue) + int(server_busy)))
            arrival_t = clock + (-mia * np.log(rng.random()))
        elif depart_t <= duration:
            clock = depart_t
            if queue:
                queue.pop(0)
                depart_t = clock + (-ms * np.log(rng.random()))
            else:
                server_busy = False
                depart_t = float('inf')
            snapshots.append((clock, len(queue) + int(server_busy)))
        else:
            break

    return snapshots


def welch_moving_average(snapshots, window):
    """Compute time-weighted moving average of number in system."""
    times  = [s[0] for s in snapshots]
    values = [s[1] for s in snapshots]
    n      = len(values)
    ma     = []
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n - 1, i + window)
        ma.append(np.mean(values[lo:hi+1]))
    return times, values, ma


def lab16_steady_state():
    print("\n" + "="*65)
    print("   LAB 16: STEADY-STATE ANALYSIS (WARM-UP DETECTION)")
    print("="*65)

    try:
        mia    = float(input("\n  Mean Inter-Arrival Time (min) [e.g. 2.0]: "))
        ms     = float(input("  Mean Service Time (min)       [e.g. 1.7]: "))
        dur    = float(input("  Simulation Duration (min)     [e.g.2000]: "))
        nreps  = int(input("  Number of replications (Welch) [e.g. 10]: "))
        win    = int(input("  Moving-average window          [e.g.  50]: "))
        seed   = int(input("  Seed                           [e.g.  42]: "))
    except ValueError:
        mia, ms, dur, nreps, win, seed = 2.0, 1.7, 2000, 10, 50, 42

    rho = ms / mia
    print(f"\n  Traffic intensity ρ = {rho:.3f}")
    if rho >= 1:
        print("  ⚠ WARNING: ρ ≥ 1 — queue unstable!")

    # ── Run R replications ──
    all_snaps = []
    for r in range(nreps):
        snaps = run_queue_sim_stream(mia, ms, dur, seed + r)
        all_snaps.append(snaps)

    print(f"\n  Ran {nreps} replications × {dur} min")

    # ── Welch's method: compute running mean per replication ──
    # Align by observation index (sample at regular time grid)
    grid    = np.linspace(0, dur, 500)
    rep_mat = np.zeros((nreps, len(grid)))

    for r, snaps in enumerate(all_snaps):
        if not snaps:
            continue
        t_arr = np.array([s[0] for s in snaps])
        v_arr = np.array([s[1] for s in snaps])
        # Interpolate to grid
        rep_mat[r] = np.interp(grid, t_arr, v_arr, left=0)

    # Average across replications for each time point
    Y_bar = np.mean(rep_mat, axis=0)

    # Moving average of Y_bar
    half = win
    Y_ma = np.convolve(Y_bar, np.ones(2*half+1)/(2*half+1), mode='same')

    # Estimate steady-state mean (last 40% of simulation)
    cutoff_idx    = int(0.6 * len(grid))
    ss_mean       = np.mean(Y_bar[cutoff_idx:])
    
    # Detect warm-up: first index where MA is within 5% of ss_mean
    tol = 0.05 * ss_mean if ss_mean > 0 else 0.1
    warmup_idx = 0
    for i, val in enumerate(Y_ma):
        if abs(val - ss_mean) < tol:
            warmup_idx = i
            break
    warmup_time = grid[warmup_idx]

    # Theoretical L = ρ/(1-ρ)
    L_th = rho / (1 - rho) if rho < 1 else float('inf')

    print(f"\n  ── Steady-State Analysis ──")
    print(tabulate([
        ["Estimated Steady-State Mean L",   f"{ss_mean:.4f}"],
        ["Theoretical L = ρ/(1-ρ)",         f"{L_th:.4f}"],
        ["Estimated Warm-Up Period",         f"{warmup_time:.1f} min"],
        ["Observations after warm-up",       f"{len(grid) - warmup_idx}"],
        ["Steady-State Mean (post warm-up)", f"{np.mean(Y_bar[warmup_idx:]):.4f}"],
    ], headers=["Metric", "Value"], tablefmt="rounded_grid"))

    # ── PLOTS ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lab 16 — Steady-State Analysis (Welch's Method)",
                 fontsize=13, fontweight='bold', color='#2c3e50')

    # A: Individual replications
    ax1 = axes[0]
    for r in range(min(nreps, 5)):
        ax1.plot(grid, rep_mat[r], alpha=0.35, linewidth=0.8)
    ax1.plot(grid, Y_bar, 'b-', linewidth=2.5, label='Mean across reps')
    ax1.axhline(L_th, color='green', linestyle='--',
                linewidth=2, label=f'Theoretical L={L_th:.2f}')
    ax1.set_title('Replications + Grand Mean', fontweight='bold')
    ax1.set_xlabel('Simulation Time (min)')
    ax1.set_ylabel('Num in System')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # B: Welch's moving average
    ax2 = axes[1]
    ax2.plot(grid, Y_bar, color='#bdc3c7', linewidth=1, label='Y̅(t)')
    ax2.plot(grid, Y_ma, color='#3498db', linewidth=2.5,
             label=f'Moving Avg (w={win})')
    ax2.axhline(ss_mean, color='#27ae60', linestyle='--',
                linewidth=2, label=f'SS Mean={ss_mean:.3f}')
    ax2.axvline(warmup_time, color='#e74c3c', linestyle=':',
                linewidth=2, label=f'Warm-up≈{warmup_time:.0f} min')
    ax2.fill_between([0, warmup_time], 0, max(Y_ma)*1.1,
                     alpha=0.1, color='#e74c3c', label='Warm-up region')
    ax2.set_title("Welch's Moving Average", fontweight='bold')
    ax2.set_xlabel('Simulation Time (min)')
    ax2.set_ylabel('Avg Num in System')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # C: Post-warmup distribution
    ax3 = axes[2]
    post_wu = Y_bar[warmup_idx:]
    ax3.hist(post_wu, bins=25, color='#2ecc71', alpha=0.8,
             edgecolor='white', density=True)
    ax3.axvline(ss_mean, color='red', linestyle='--',
                linewidth=2, label=f'SS Mean={ss_mean:.3f}')
    ax3.axvline(L_th, color='green', linestyle=':',
                linewidth=2, label=f'Theory={L_th:.3f}')
    ax3.set_title('Post-Warmup Distribution of L', fontweight='bold')
    ax3.set_xlabel('Num in System')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab16_steady_state.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab16_steady_state.png")
    plt.show()
    print("\n✅ Lab 16 Complete!\n")


# ═══════════════════════════════════════════════════════════
# LAB 17 — BANK ATM SIMULATION
# ═══════════════════════════════════════════════════════════

def erlang_c(c, lam, mu):
    """
    Erlang-C formula: P(wait > 0) for M/M/c queue.
    c = servers, lam = arrival rate, mu = service rate per server
    """
    rho = lam / (c * mu)
    if rho >= 1:
        return 1.0
    A = lam / mu  # offered traffic
    # Numerator: A^c / c! × 1/(1-rho)
    numer = (A**c / math.factorial(c)) * (1 / (1 - rho))
    # Denominator: sum_{k=0}^{c-1} A^k/k!  +  numer
    denom = sum(A**k / math.factorial(k) for k in range(c)) + numer
    return numer / denom


def mm_c_metrics(c, lam, mu):
    """Compute M/M/c theoretical metrics."""
    rho  = lam / (c * mu)
    C    = erlang_c(c, lam, mu)
    Wq   = C / (c * mu - lam)
    Lq   = lam * Wq
    W    = Wq + 1/mu
    L    = lam * W
    return {'rho': rho, 'C': C, 'Wq': Wq, 'Lq': Lq, 'W': W, 'L': L}


def simulate_atm(n_atms, lam, mu, duration, seed=42):
    """
    Event-driven M/M/c simulation.
    Returns avg wait, max queue, utilization.
    """
    rng         = np.random.default_rng(seed)
    atm_free    = [True] * n_atms   # True = idle
    queue       = []                  # queue of arrival times
    clock       = 0.0
    departure_t = [float('inf')] * n_atms
    total_wait  = 0.0
    served      = 0
    max_queue   = 0
    busy_area   = [0.0] * n_atms
    last_t      = 0.0

    arrival_t = -1/lam * np.log(rng.random())
    queue_len_hist = []

    while clock < duration:
        # Next event candidates
        candidates = [(arrival_t, 'arrival', -1)]
        for i, dt in enumerate(departure_t):
            if dt < float('inf'):
                candidates.append((dt, 'departure', i))
        candidates.sort()
        t_next, etype, idx = candidates[0]

        if t_next > duration:
            break

        # Accumulate busy area
        dt_pass = t_next - last_t
        for i in range(n_atms):
            if not atm_free[i]:
                busy_area[i] += dt_pass
        last_t = t_next
        clock  = t_next
        queue_len_hist.append((clock, len(queue)))

        if etype == 'arrival':
            free_atm = next((i for i in range(n_atms) if atm_free[i]), None)
            if free_atm is not None:
                atm_free[free_atm]    = False
                svc_t                 = -1/mu * np.log(rng.random())
                departure_t[free_atm] = clock + svc_t
                total_wait += 0
            else:
                queue.append(clock)
                max_queue = max(max_queue, len(queue))
            arrival_t = clock + (-1/lam * np.log(rng.random()))
            served += 1

        elif etype == 'departure':
            if queue:
                arrived_at  = queue.pop(0)
                wait        = clock - arrived_at
                total_wait += wait
                svc_t       = -1/mu * np.log(rng.random())
                departure_t[idx] = clock + svc_t
            else:
                atm_free[idx]    = True
                departure_t[idx] = float('inf')

    avg_util = [ba / duration for ba in busy_area]
    avg_wait = total_wait / max(served, 1)
    return avg_wait, max_queue, avg_util, queue_len_hist


def lab17_atm():
    print("\n" + "="*65)
    print("   LAB 17: BANK ATM SYSTEM SIMULATION (M/M/c)")
    print("="*65)

    try:
        n_atms   = int(input("\n  Number of ATMs (c)              [e.g.   3]: "))
        lam      = float(input("  Arrival rate λ (customers/min)  [e.g. 1.5]: "))
        mu       = float(input("  Service rate μ (customers/min)  [e.g. 1.0]: "))
        duration = float(input("  Simulation duration (min)       [e.g.1000]: "))
        seed     = int(input("  Random seed                     [e.g.  42]: "))
    except ValueError:
        n_atms, lam, mu, duration, seed = 3, 1.5, 1.0, 1000, 42

    rho_sys = lam / (n_atms * mu)
    print(f"\n  Traffic intensity ρ = λ/(cμ) = {rho_sys:.3f}")
    if rho_sys >= 1:
        print("  ⚠ WARNING: ρ ≥ 1 → queue unstable!")

    # ── Theoretical (M/M/c) ──
    th = mm_c_metrics(n_atms, lam, mu)

    # ── Simulation ──
    avg_wait, max_q, avg_util, q_hist = simulate_atm(
        n_atms, lam, mu, duration, seed)

    print("\n  ── M/M/c Results ──")
    print(tabulate([
        ["Server utilization ρ",           f"{th['rho']:.4f}",  f"{np.mean(avg_util):.4f}"],
        ["P(customer waits) [Erlang-C]",   f"{th['C']:.4f}",    "—"],
        ["Avg wait in queue Wq (min)",      f"{th['Wq']:.4f}",   f"{avg_wait:.4f}"],
        ["Avg queue length Lq",             f"{th['Lq']:.4f}",   "—"],
        ["Avg time in system W (min)",      f"{th['W']:.4f}",    "—"],
        ["Avg number in system L",          f"{th['L']:.4f}",    "—"],
        ["Max queue length observed",       "—",                  str(max_q)],
    ], headers=["Metric", "Theoretical", "Simulated"],
       tablefmt="rounded_grid"))

    # ATM utilization table
    print("\n  ── Per-ATM Utilization ──")
    atm_rows = [(f"ATM {i+1}", f"{u:.4f}", f"{u*100:.1f}%")
                for i, u in enumerate(avg_util)]
    print(tabulate(atm_rows,
                   headers=["ATM", "Utilization", "%"],
                   tablefmt="rounded_grid"))

    # Compare different numbers of ATMs
    print("\n  ── Sensitivity: Number of ATMs ──")
    sens_rows = []
    for c in range(1, n_atms + 3):
        if lam / (c * mu) >= 1:
            sens_rows.append([c, "UNSTABLE", "—", "—", "—"])
        else:
            m = mm_c_metrics(c, lam, mu)
            sens_rows.append([c,
                              f"{m['rho']:.3f}",
                              f"{m['C']:.3f}",
                              f"{m['Wq']:.3f}",
                              f"{m['Lq']:.3f}"])
    print(tabulate(sens_rows,
                   headers=["# ATMs", "ρ", "P(wait)", "Wq (min)", "Lq"],
                   tablefmt="rounded_grid"))

    # ── PLOTS ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Lab 17 — Bank ATM Simulation (c={n_atms} ATMs, λ={lam}, μ={mu})",
                 fontsize=13, fontweight='bold', color='#2c3e50')

    # A: Queue length over time
    ax1 = axes[0, 0]
    if q_hist:
        qt = [h[0] for h in q_hist]
        qv = [h[1] for h in q_hist]
        ax1.step(qt, qv, where='post', color='#3498db', linewidth=1.2)
        ax1.fill_between(qt, qv, step='post', alpha=0.2, color='#3498db')
        ax1.axhline(th['Lq'], color='red', linestyle='--',
                    linewidth=2, label=f'Theory Lq={th["Lq"]:.2f}')
    ax1.set_title('Queue Length Over Time', fontweight='bold')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Queue Length')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # B: ATM utilization bar
    ax2 = axes[0, 1]
    colors = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']
    atm_labels = [f"ATM {i+1}" for i in range(n_atms)]
    bars = ax2.bar(atm_labels, [u*100 for u in avg_util],
                   color=colors[:n_atms], edgecolor='white')
    ax2.bar_label(bars, fmt='%.1f%%', fontsize=9)
    ax2.axhline(th['rho']*100, color='red', linestyle='--',
                linewidth=2, label=f'Theory ρ={th["rho"]*100:.1f}%')
    ax2.set_title('ATM Utilization (%)', fontweight='bold')
    ax2.set_ylabel('Utilization (%)')
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#f8f9fa')

    # C: Sensitivity — Wq vs # ATMs
    ax3 = axes[1, 0]
    c_vals = list(range(1, n_atms + 5))
    wq_vals, lq_vals = [], []
    for c in c_vals:
        if lam / (c * mu) < 1:
            m = mm_c_metrics(c, lam, mu)
            wq_vals.append(m['Wq'])
            lq_vals.append(m['Lq'])
        else:
            wq_vals.append(None)
            lq_vals.append(None)
    valid  = [(c, w) for c, w in zip(c_vals, wq_vals) if w is not None]
    cv, wv = zip(*valid) if valid else ([], [])
    ax3.plot(cv, wv, 'o-', color='#3498db', linewidth=2.5, markersize=8)
    ax3.axvline(n_atms, color='red', linestyle='--',
                linewidth=2, label=f'Current c={n_atms}')
    ax3.set_title('Avg Wait Time vs # ATMs', fontweight='bold')
    ax3.set_xlabel('Number of ATMs (c)')
    ax3.set_ylabel('Wq (min)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')

    # D: Erlang-C (P wait) vs load
    ax4 = axes[1, 1]
    rho_range = np.linspace(0.1, 0.99, 100)
    for c in [1, 2, 3, 4]:
        ec_vals = []
        for r in rho_range:
            lam_t = r * c * mu
            try:
                ec_vals.append(erlang_c(c, lam_t, mu))
            except:
                ec_vals.append(1.0)
        ax4.plot(rho_range, ec_vals, linewidth=2, label=f'c={c}')
    ax4.axvline(rho_sys, color='red', linestyle='--',
                linewidth=2, label=f'Current ρ={rho_sys:.2f}')
    ax4.set_title('Erlang-C: P(Wait) vs Traffic Load', fontweight='bold')
    ax4.set_xlabel('Traffic Intensity ρ')
    ax4.set_ylabel('P(Customer Waits)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab17_atm_simulation.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab17_atm_simulation.png")
    plt.show()
    print("\n✅ Lab 17 Complete!\n")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  CIT 324 — Lab 16 & 17                          ║")
    print("╚══════════════════════════════════════════════════╝")
    print("\n  [16] Lab 16 — Steady-State Analysis")
    print("  [17] Lab 17 — Bank ATM Simulation")
    print("  [0]  Run BOTH")

    try:
        ch = int(input("\n  Choice [0/16/17]: "))
    except ValueError:
        ch = 0

    if ch == 16 or ch == 0:
        lab16_steady_state()
    if ch == 17 or ch == 0:
        lab17_atm()


if __name__ == "__main__":
    main()
