"""
=============================================================
CIT 324 - Simulation and Modeling Sessional
Lab 01: M/M/1 Queue Simulation
=============================================================
Theory (Averill M. Law, Chapter 1 & 2):
  - M/M/1 queue: Single server, FIFO discipline
  - Arrival: Poisson process => Inter-arrival time ~ Exponential(λ)
  - Service: ~ Exponential(μ)
  - Key measures: Avg delay, Avg queue length, Server utilization
  - Event-driven simulation using next-event time advance
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tabulate import tabulate

# ─────────────────────────────────────────
# RANDOM NUMBER GENERATION (Exponential)
# Using Inverse Transform: X = -mean * ln(U)
# (Law textbook, Section 8.3.2)
# ─────────────────────────────────────────
def exponential(mean, rng):
    U = rng.random()
    return -mean * np.log(U)


# ─────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────
def mm1_simulation(mean_interarrival, mean_service, max_customers, seed=42):
    """
    M/M/1 queue simulation using next-event time advance.

    Parameters:
        mean_interarrival : average time between arrivals (1/λ)
        mean_service      : average service time (1/μ)
        max_customers     : stop after this many customers served
        seed              : RNG seed for reproducibility

    Returns:
        dict of performance metrics + trace data
    """
    rng = np.random.default_rng(seed)

    # ── System State Variables ──
    sim_clock       = 0.0          # current simulation time
    num_in_queue    = 0            # customers waiting in queue
    server_busy     = False        # server status
    num_served      = 0            # total customers served

    # ── Statistical Accumulators ──
    total_delay       = 0.0        # sum of all delays in queue
    area_num_in_queue = 0.0        # area under queue-length curve (for avg Lq)
    area_server_busy  = 0.0        # area under server-busy curve (for utilization)

    # ── Event Calendar ──
    arrival_time     = exponential(mean_interarrival, rng)  # first arrival
    departure_time   = float('inf')                          # no departure yet

    # ── Queued Arrival Times (FIFO) ──
    queue = []   # stores time each waiting customer arrived

    # ── Trace Log ──
    trace = []   # for detailed event log display

    last_event_time = 0.0

    print("\n" + "="*65)
    print("   M/M/1 QUEUE SIMULATION  (Next-Event Time Advance)")
    print("="*65)
    print(f"   Mean Inter-Arrival Time : {mean_interarrival} min")
    print(f"   Mean Service Time       : {mean_service} min")
    print(f"   Max Customers           : {max_customers}")
    print(f"   Traffic Intensity ρ     : {mean_service/mean_interarrival:.3f}")
    print("="*65)

    # ─────────────────────────────────────────
    # MAIN SIMULATION LOOP
    # ─────────────────────────────────────────
    while num_served < max_customers:

        # ── Determine next event (arrival or departure) ──
        if arrival_time <= departure_time:
            # ── ARRIVAL EVENT ──
            dt = arrival_time - last_event_time
            area_num_in_queue += num_in_queue * dt
            area_server_busy  += server_busy * dt
            last_event_time = arrival_time
            sim_clock = arrival_time

            if not server_busy:
                # Server is idle → customer goes straight to service
                server_busy = True
                delay = 0.0
                total_delay += delay
                departure_time = sim_clock + exponential(mean_service, rng)
            else:
                # Server busy → customer joins queue
                queue.append(sim_clock)
                num_in_queue += 1

            trace.append({
                'time'   : round(sim_clock, 4),
                'event'  : 'ARRIVAL',
                'queue'  : num_in_queue,
                'server' : 'BUSY',
            })

            # Schedule next arrival
            arrival_time = sim_clock + exponential(mean_interarrival, rng)

        else:
            # ── DEPARTURE EVENT ──
            dt = departure_time - last_event_time
            area_num_in_queue += num_in_queue * dt
            area_server_busy  += server_busy * dt
            last_event_time = departure_time
            sim_clock = departure_time
            num_served += 1

            if num_in_queue > 0:
                # Take next customer from queue (FIFO)
                waited_since = queue.pop(0)
                num_in_queue -= 1
                delay = sim_clock - waited_since
                total_delay += delay
                departure_time = sim_clock + exponential(mean_service, rng)
            else:
                # Queue empty → server goes idle
                server_busy = False
                departure_time = float('inf')

            trace.append({
                'time'   : round(sim_clock, 4),
                'event'  : 'DEPARTURE',
                'queue'  : num_in_queue,
                'server' : 'BUSY' if server_busy else 'IDLE',
            })

    # ─────────────────────────────────────────
    # PERFORMANCE MEASURES
    # (Law textbook, Chapter 1 equations)
    # ─────────────────────────────────────────
    avg_delay       = total_delay / max_customers                    # W_q
    avg_num_queue   = area_num_in_queue / sim_clock                  # L_q
    server_util     = area_server_busy / sim_clock                   # ρ̂
    avg_time_system = avg_delay + mean_service                       # W = Wq + 1/μ
    avg_num_system  = avg_time_system * (1 / mean_interarrival)      # L = λW

    # Theoretical values (M/M/1 formulas)
    lam   = 1 / mean_interarrival
    mu    = 1 / mean_service
    rho   = lam / mu
    Lq_th = rho**2 / (1 - rho) if rho < 1 else float('inf')
    Wq_th = Lq_th / lam
    L_th  = rho / (1 - rho) if rho < 1 else float('inf')
    W_th  = L_th / lam

    return {
        'avg_delay'       : avg_delay,
        'avg_num_queue'   : avg_num_queue,
        'server_util'     : server_util,
        'sim_time'        : sim_clock,
        'avg_time_system' : avg_time_system,
        'avg_num_system'  : avg_num_system,
        'theoretical'     : {
            'rho' : rho, 'Lq' : Lq_th, 'Wq' : Wq_th,
            'L'   : L_th, 'W'  : W_th,
        },
        'trace': trace,
    }


# ─────────────────────────────────────────
# OUTPUT & PLOTTING
# ─────────────────────────────────────────
def print_results(res):
    th = res['theoretical']
    print("\n" + "─"*65)
    print("   SIMULATION RESULTS")
    print("─"*65)

    rows = [
        ["Average Delay in Queue (Wq)",
            f"{res['avg_delay']:.4f} min",
            f"{th['Wq']:.4f} min"],
        ["Average Number in Queue (Lq)",
            f"{res['avg_num_queue']:.4f}",
            f"{th['Lq']:.4f}"],
        ["Server Utilization (ρ)",
            f"{res['server_util']:.4f}",
            f"{th['rho']:.4f}"],
        ["Avg Time in System (W)",
            f"{res['avg_time_system']:.4f} min",
            f"{th['W']:.4f} min"],
        ["Avg Number in System (L)",
            f"{res['avg_num_system']:.4f}",
            f"{th['L']:.4f}"],
        ["Time Simulation Ended",
            f"{res['sim_time']:.4f} min", "—"],
    ]
    print(tabulate(rows,
                   headers=["Metric", "Simulated", "Theoretical (M/M/1)"],
                   tablefmt="rounded_grid"))

    # Print first 15 events as trace
    print("\n── Event Trace (first 15 events) ──")
    trace_rows = [(e['time'], e['event'], e['queue'], e['server'])
                  for e in res['trace'][:15]]
    print(tabulate(trace_rows,
                   headers=["Sim Time", "Event", "Queue Length", "Server"],
                   tablefmt="rounded_grid"))


def plot_results(results_list, labels):
    """
    Plots comparing simulated vs theoretical across different traffic loads.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("M/M/1 Queue Simulation — Results Dashboard",
                 fontsize=15, fontweight='bold', color='#2c3e50')

    colors_sim = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    colors_th  = ['#1a5276', '#922b21', '#1e8449', '#b7770d']

    metrics_sim = ['avg_delay', 'avg_num_queue', 'server_util', 'avg_time_system']
    metrics_th  = ['Wq', 'Lq', 'rho', 'W']
    titles      = ['Avg Delay in Queue (Wq)',
                   'Avg Number in Queue (Lq)',
                   'Server Utilization (ρ)',
                   'Avg Time in System (W)']
    ylabels     = ['Minutes', 'Customers', 'Utilization', 'Minutes']

    rhos = [r['theoretical']['rho'] for r in results_list]

    for ax, ms, mt, title, ylabel in zip(axes.flat, metrics_sim,
                                          metrics_th, titles, ylabels):
        sim_vals = [r[ms] for r in results_list]
        th_vals  = [r['theoretical'][mt] for r in results_list]

        ax.plot(rhos, sim_vals, 'o-', color='#3498db',
                linewidth=2, markersize=7, label='Simulated')
        ax.plot(rhos, th_vals, 's--', color='#e74c3c',
                linewidth=2, markersize=7, label='Theoretical')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Traffic Intensity ρ = λ/μ', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab01_mm1_results.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab01_mm1_results.png")
    plt.show()


def plot_queue_trace(res, max_events=80):
    """Queue length over time (event trace plot)."""
    times  = [0]
    queues = [0]
    for e in res['trace'][:max_events]:
        times.append(e['time'])
        queues.append(e['queue'])

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.step(times, queues, where='post', color='#3498db', linewidth=1.8)
    ax.fill_between(times, queues, step='post', alpha=0.25, color='#3498db')
    ax.set_title('Queue Length Over Simulation Time (first 80 events)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Simulation Time (min)')
    ax.set_ylabel('Number in Queue')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab01_queue_trace.png',
                dpi=150, bbox_inches='tight')
    print("✅ Plot saved: lab01_queue_trace.png")
    plt.show()


# ─────────────────────────────────────────
# MAIN — USER INPUT
# ─────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════════╗")
    print("║   CIT 324 — Lab 01: M/M/1 Queue Sim     ║")
    print("╚══════════════════════════════════════════╝")

    try:
        mia = float(input("\n  Enter Mean Inter-Arrival Time (min) [e.g. 2.0]: "))
        ms  = float(input("  Enter Mean Service Time (min)       [e.g. 1.5]: "))
        mc  = int(input("  Enter Max Customers to simulate    [e.g. 500]: "))
        seed = int(input("  Enter Random Seed                  [e.g. 42]  : "))
    except ValueError:
        print("  ⚠ Invalid input. Using defaults: λ=0.5, μ=0.667, N=1000")
        mia, ms, mc, seed = 2.0, 1.5, 1000, 42

    rho = ms / mia
    if rho >= 1.0:
        print(f"\n  ⚠ WARNING: ρ = {rho:.3f} ≥ 1 → queue is UNSTABLE (infinite growth)")

    # Run primary simulation
    res = mm1_simulation(mia, ms, mc, seed)
    print_results(res)
    plot_queue_trace(res)

    # ── Sensitivity: vary ρ from 0.2 to 0.9 ──
    print("\n── Running sensitivity analysis across different ρ values... ──")
    rho_values     = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
    results_list   = []
    labels         = []
    for rho_target in rho_values:
        ia = 2.0
        sv = rho_target * ia  # service time to hit target ρ
        r  = mm1_simulation(ia, sv, mc, seed)
        results_list.append(r)
        labels.append(f"ρ={rho_target}")

    plot_results(results_list, labels)
    print("\n✅ Lab 01 Complete!\n")


if __name__ == "__main__":
    main()
