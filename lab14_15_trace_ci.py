"""
=============================================================
CIT 324 — Lab 14: Debugging & Traceability (DES Verification)
         Lab 15: Confidence Interval Estimation
=============================================================
Theory (Averill M. Law, Chapters 5 & 9):

  Lab 14 — Verification:
    Step through a DES and log every state change.
    Verification ensures the model matches the LOGICAL DESIGN.
    Checks: clock advances correctly, queue discipline is FIFO,
    server status consistent, no negative delays.
    A "trace" program prints every event to catch bugs.

  Lab 15 — Confidence Interval (Law, Chapter 9):
    After R independent replications:
      Point estimate: X̄ = (1/R) Σ Xᵢ
      Sample variance: S² = Σ(Xᵢ - X̄)²/(R-1)
      Standard error: SE = S/√R
      95% CI: X̄ ± t_{R-1, 0.025} × SE
    Half-width: h = t_{R-1, 0.025} × SE
    Required replications for precision ε:
      R* = ceil( (t_{R-1,0.025} × S / ε)² )
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from tabulate import tabulate
import math


# ═══════════════════════════════════════════════════════════
# LAB 14 — DES TRACE / VERIFICATION
# ═══════════════════════════════════════════════════════════

def lab14_trace_verification():
    print("\n" + "="*65)
    print("   LAB 14: DEBUGGING & TRACEABILITY (DES VERIFICATION)")
    print("="*65)

    try:
        mia  = float(input("\n  Mean Inter-Arrival Time (min) [e.g. 3.0]: "))
        ms   = float(input("  Mean Service Time (min)       [e.g. 2.0]: "))
        nc   = int(input("  Number of customers to trace  [e.g.  15]: "))
        seed = int(input("  Random Seed                   [e.g.  42]: "))
    except ValueError:
        mia, ms, nc, seed = 3.0, 2.0, 15, 42

    rng = np.random.default_rng(seed)

    print(f"\n  ── Simulation Parameters ──")
    print(f"  Mean Inter-Arrival: {mia} min   (λ = {1/mia:.4f}/min)")
    print(f"  Mean Service Time : {ms} min    (μ = {1/ms:.4f}/min)")
    print(f"  Traffic Intensity : ρ = {ms/mia:.3f}")

    # ── Internal State ──
    clock         = 0.0
    server_busy   = False
    num_in_queue  = 0
    queue_times   = []       # arrival times of queued customers
    events_log    = []       # full trace log
    checks_log    = []       # verification checks

    arrival_t   = -mia * np.log(rng.random())
    departure_t = float('inf')
    num_served  = 0

    total_delay  = 0.0
    busy_area    = 0.0
    queue_area   = 0.0
    last_t       = 0.0

    def check(cond, msg):
        """Verification assertion."""
        status = "✅ OK " if cond else "❌ ERR"
        checks_log.append((status, msg))
        if not cond:
            print(f"  ⚠ VERIFICATION FAILED: {msg}")

    while num_served < nc:
        if arrival_t <= departure_t:
            # ARRIVAL
            dt = arrival_t - last_t
            busy_area  += int(server_busy) * dt
            queue_area += num_in_queue * dt
            last_t      = arrival_t
            clock       = arrival_t

            prev_status = "BUSY" if server_busy else "IDLE"

            if not server_busy:
                server_busy   = True
                delay         = 0.0
                svc           = -ms * np.log(rng.random())
                departure_t   = clock + svc
                total_delay  += delay
                check(delay >= 0, f"t={clock:.3f}: delay ≥ 0")
                check(svc > 0,    f"t={clock:.3f}: service time > 0")
            else:
                queue_times.append(clock)
                num_in_queue += 1
                delay = None
                check(num_in_queue > 0, f"t={clock:.3f}: queue count positive")

            next_ia   = -mia * np.log(rng.random())
            arrival_t = clock + next_ia
            check(next_ia > 0, f"t={clock:.3f}: inter-arrival > 0")

            events_log.append({
                'clock'    : clock,
                'event'    : 'ARRIVAL',
                'queue'    : num_in_queue,
                'server'   : 'BUSY',
                'delay'    : f"{delay:.4f}" if delay is not None else "—",
                'next_arr' : round(arrival_t, 4),
                'next_dep' : round(departure_t, 4) if departure_t < 1e9 else "—",
            })

        else:
            # DEPARTURE
            dt = departure_t - last_t
            busy_area  += int(server_busy) * dt
            queue_area += num_in_queue * dt
            last_t      = departure_t
            clock       = departure_t
            num_served += 1

            check(server_busy, f"t={clock:.3f}: server must be BUSY at departure")

            if num_in_queue > 0:
                waited_since  = queue_times.pop(0)   # FIFO
                num_in_queue -= 1
                delay         = clock - waited_since
                total_delay  += delay
                svc           = -ms * np.log(rng.random())
                departure_t   = clock + svc
                check(delay >= 0,     f"t={clock:.3f}: waiting delay ≥ 0")
                check(num_in_queue >= 0, f"t={clock:.3f}: queue ≥ 0")
            else:
                server_busy = False
                departure_t = float('inf')
                delay       = 0.0

            events_log.append({
                'clock'    : clock,
                'event'    : 'DEPARTURE',
                'queue'    : num_in_queue,
                'server'   : 'BUSY' if server_busy else 'IDLE',
                'delay'    : f"{delay:.4f}",
                'next_arr' : round(arrival_t, 4),
                'next_dep' : round(departure_t, 4) if departure_t < 1e9 else "—",
            })

    # ── Print Trace Table ──
    print("\n  ── FULL EVENT TRACE ──")
    trace_rows = [
        (f"{e['clock']:.4f}", e['event'], e['queue'],
         e['server'], e['delay'], e['next_arr'], e['next_dep'])
        for e in events_log
    ]
    print(tabulate(trace_rows,
                   headers=["Clock", "Event", "Queue",
                             "Server", "Delay", "Next Arrival", "Next Depart"],
                   tablefmt="rounded_grid",
                   floatfmt=".4f"))

    # ── Performance (simulated) ──
    avg_delay  = total_delay / num_served
    util       = busy_area / clock
    avg_queue  = queue_area / clock

    # Theoretical M/M/1
    rho = ms / mia
    Wq_th = rho / ((1/ms) * (1 - rho))
    Lq_th = rho**2 / (1 - rho)

    print("\n  ── Performance Measures ──")
    print(tabulate([
        ["Avg Delay in Queue (Wq)", f"{avg_delay:.4f}", f"{Wq_th:.4f}"],
        ["Server Utilization (ρ)",  f"{util:.4f}",      f"{rho:.4f}"],
        ["Avg Queue Length (Lq)",   f"{avg_queue:.4f}", f"{Lq_th:.4f}"],
    ], headers=["Metric", "Simulated", "Theoretical (M/M/1)"],
       tablefmt="rounded_grid"))

    # ── Verification Checks Summary ──
    print(f"\n  ── Verification Checks ({len(checks_log)} total) ──")
    passed = sum(1 for s, _ in checks_log if "OK" in s)
    failed = len(checks_log) - passed
    print(f"  ✅ Passed: {passed}   ❌ Failed: {failed}")
    if failed > 0:
        for status, msg in checks_log:
            if "ERR" in status:
                print(f"   → {status}: {msg}")
    else:
        print("  All assertions passed — model logic verified ✓")

    # ── PLOTS ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lab 14 — DES Trace & Verification",
                 fontsize=13, fontweight='bold', color='#2c3e50')

    clocks = [e['clock']    for e in events_log]
    queues = [e['queue']    for e in events_log]
    srvrs  = [1 if e['server'] == 'BUSY' else 0 for e in events_log]

    # A: Queue over time
    ax1 = axes[0]
    ax1.step(clocks, queues, where='post', color='#3498db', linewidth=2)
    ax1.fill_between(clocks, queues, step='post', alpha=0.2, color='#3498db')
    ax1.set_title('Queue Length Over Time', fontweight='bold')
    ax1.set_xlabel('Simulation Time (min)')
    ax1.set_ylabel('Queue Length')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # B: Server status
    ax2 = axes[1]
    ax2.step(clocks, srvrs, where='post', color='#e74c3c', linewidth=2)
    ax2.fill_between(clocks, srvrs, step='post', alpha=0.2, color='#e74c3c')
    ax2.set_title('Server Status (1=Busy, 0=Idle)', fontweight='bold')
    ax2.set_xlabel('Simulation Time (min)')
    ax2.set_ylabel('Server Status')
    ax2.set_ylim(-0.1, 1.3)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # C: Verification bar chart
    ax3 = axes[2]
    ax3.bar(['Passed', 'Failed'], [passed, failed],
            color=['#27ae60', '#e74c3c'], edgecolor='white', width=0.4)
    ax3.set_title(f'Verification Results\n({len(checks_log)} checks)',
                  fontweight='bold')
    ax3.set_ylabel('Count')
    for i, v in enumerate([passed, failed]):
        ax3.text(i, v + 0.3, str(v), ha='center', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab14_trace_verification.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab14_trace_verification.png")
    plt.show()
    print("\n✅ Lab 14 Complete!\n")


# ═══════════════════════════════════════════════════════════
# LAB 15 — CONFIDENCE INTERVAL ESTIMATION
# ═══════════════════════════════════════════════════════════

def run_one_replication(mia, ms, n_customers, rng):
    """Run one M/M/1 replication, return avg waiting time."""
    server_busy = False
    queue       = []
    arrival_t   = -mia * np.log(rng.random())
    depart_t    = float('inf')
    total_delay = 0.0
    served      = 0

    while served < n_customers:
        if arrival_t <= depart_t:
            if not server_busy:
                server_busy = True
                total_delay += 0
                depart_t = arrival_t + (-ms * np.log(rng.random()))
            else:
                queue.append(arrival_t)
            arrival_t += -mia * np.log(rng.random())
        else:
            served += 1
            if queue:
                waited_since = queue.pop(0)
                total_delay += depart_t - waited_since
                depart_t += -ms * np.log(rng.random())
            else:
                server_busy = False
                depart_t = float('inf')

    return total_delay / n_customers


def lab15_confidence_interval():
    print("\n" + "="*65)
    print("   LAB 15: CONFIDENCE INTERVAL ESTIMATION")
    print("="*65)

    try:
        mia    = float(input("\n  Mean Inter-Arrival Time (min)     [e.g. 2.0]: "))
        ms     = float(input("  Mean Service Time (min)           [e.g. 1.5]: "))
        nc     = int(input("  Customers per replication         [e.g. 200]: "))
        R      = int(input("  Number of replications            [e.g.  20]: "))
        alpha  = float(input("  Significance level α (e.g. 0.05)  [e.g.0.05]: "))
        eps    = float(input("  Desired half-width precision ε    [e.g. 0.2]: "))
        seed   = int(input("  Random seed                       [e.g.  42]: "))
    except ValueError:
        mia, ms, nc, R, alpha, eps, seed = 2.0, 1.5, 200, 20, 0.05, 0.2, 42

    rng = np.random.default_rng(seed)

    print(f"\n  Running {R} independent replications...")
    obs = [run_one_replication(mia, ms, nc, rng) for _ in range(R)]

    # ── Point Estimate & Variance ──
    X_bar = np.mean(obs)
    S2    = np.var(obs, ddof=1)
    S     = np.sqrt(S2)
    SE    = S / np.sqrt(R)

    # t critical value
    t_crit = stats.t.ppf(1 - alpha / 2, df=R - 1)
    half_w = t_crit * SE
    CI_lo  = X_bar - half_w
    CI_hi  = X_bar + half_w

    # Theoretical Wq for M/M/1
    rho    = ms / mia
    Wq_th  = rho / ((1/ms) * (1 - rho)) if rho < 1 else float('inf')

    # Required replications for desired precision ε
    R_star = math.ceil((t_crit * S / eps) ** 2)

    # ── Print Results ──
    print(f"\n  ── Replication Results ──")
    rep_rows = [(i+1, f"{v:.4f}") for i, v in enumerate(obs)]
    print(tabulate(rep_rows, headers=["Replication", "Avg Waiting Time (min)"],
                   tablefmt="rounded_grid"))

    print(f"\n  ── Statistical Analysis ──")
    print(tabulate([
        ["Point Estimate X̄",            f"{X_bar:.4f} min"],
        ["Sample Variance S²",           f"{S2:.4f}"],
        ["Sample Std Dev S",             f"{S:.4f}"],
        ["Standard Error SE = S/√R",     f"{SE:.4f}"],
        ["t_{R-1, α/2}",                 f"{t_crit:.4f}  (df={R-1}, α={alpha})"],
        ["Half-Width h",                 f"{half_w:.4f} min"],
        ["─"*30,                          "─"*20],
        [f"95% CI",                      f"[{CI_lo:.4f},  {CI_hi:.4f}]"],
        ["─"*30,                          "─"*20],
        ["Theoretical Wq (M/M/1)",       f"{Wq_th:.4f} min"],
        ["CI contains theoretical?",     "✅ Yes" if CI_lo <= Wq_th <= CI_hi else "❌ No"],
        ["─"*30,                          "─"*20],
        [f"Required reps for ε={eps}",   f"{R_star} replications"],
    ], headers=["Metric", "Value"], tablefmt="rounded_grid"))

    # ── PLOTS ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lab 15 — Confidence Interval Estimation",
                 fontsize=13, fontweight='bold', color='#2c3e50')

    reps = list(range(1, R + 1))

    # A: Replication values + CI band
    ax1 = axes[0]
    ax1.plot(reps, obs, 'o-', color='#3498db', linewidth=1.5,
             markersize=6, label='Replication Wq')
    ax1.axhline(X_bar, color='#e74c3c', linewidth=2.5,
                linestyle='-', label=f'X̄ = {X_bar:.3f}')
    ax1.axhline(CI_lo, color='#e74c3c', linewidth=1.5,
                linestyle='--', label=f'95% CI')
    ax1.axhline(CI_hi, color='#e74c3c', linewidth=1.5, linestyle='--')
    ax1.axhline(Wq_th, color='#27ae60', linewidth=2,
                linestyle=':', label=f'Theory Wq={Wq_th:.3f}')
    ax1.fill_between([1, R], CI_lo, CI_hi, alpha=0.1, color='#e74c3c')
    ax1.set_title('Replication Observations', fontweight='bold')
    ax1.set_xlabel('Replication')
    ax1.set_ylabel('Avg Waiting Time (min)')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # B: Running mean convergence
    ax2 = axes[1]
    run_mean = np.cumsum(obs) / np.arange(1, R + 1)
    run_se   = [np.std(obs[:i+1], ddof=1)/np.sqrt(i+1) if i > 0 else 0
                for i in range(R)]
    t_vals   = [stats.t.ppf(0.975, df=max(i, 1)) for i in range(R)]
    hw_run   = [t * se for t, se in zip(t_vals, run_se)]
    ax2.plot(reps, run_mean, 'b-', linewidth=2, label='Running Mean')
    ax2.fill_between(reps,
                     [m - h for m, h in zip(run_mean, hw_run)],
                     [m + h for m, h in zip(run_mean, hw_run)],
                     alpha=0.2, color='blue', label='95% CI band')
    ax2.axhline(Wq_th, color='green', linestyle=':', linewidth=2,
                label=f'Theory={Wq_th:.3f}')
    ax2.set_title('Convergence of Running Mean', fontweight='bold')
    ax2.set_xlabel('Replications')
    ax2.set_ylabel('Mean Waiting Time')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # C: Sample distribution histogram
    ax3 = axes[2]
    ax3.hist(obs, bins=10, color='#2ecc71', alpha=0.8,
             edgecolor='white', density=True)
    xn = np.linspace(min(obs), max(obs), 200)
    ax3.plot(xn, stats.norm.pdf(xn, X_bar, S), 'r-',
             linewidth=2.5, label='Normal fit')
    ax3.axvline(X_bar, color='red', linestyle='--', linewidth=2,
                label=f'Mean={X_bar:.3f}')
    ax3.axvline(CI_lo, color='orange', linestyle=':', linewidth=2,
                label=f'CI [{CI_lo:.3f}, {CI_hi:.3f}]')
    ax3.axvline(CI_hi, color='orange', linestyle=':', linewidth=2)
    ax3.set_title('Distribution of Replication Means', fontweight='bold')
    ax3.set_xlabel('Avg Waiting Time')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab15_confidence_interval.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab15_confidence_interval.png")
    plt.show()
    print("\n✅ Lab 15 Complete!\n")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  CIT 324 — Lab 14 & 15                          ║")
    print("╚══════════════════════════════════════════════════╝")
    print("\n  [14] Lab 14 — DES Trace & Verification")
    print("  [15] Lab 15 — Confidence Interval")
    print("  [0]  Run BOTH")

    try:
        ch = int(input("\n  Choice [0/14/15]: "))
    except ValueError:
        ch = 0

    if ch == 14 or ch == 0:
        lab14_trace_verification()
    if ch == 15 or ch == 0:
        lab15_confidence_interval()


if __name__ == "__main__":
    main()
