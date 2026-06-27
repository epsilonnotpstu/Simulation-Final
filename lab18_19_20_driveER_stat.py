"""
=============================================================
CIT 324 — Lab 18: Fast-Food Drive-Thru Bottleneck Analysis
         Lab 19: Emergency Room (ER) Patient Flow
         Lab 20: Statistical Output Analysis — Drive-Thru Pharmacy
=============================================================
Theory (Averill M. Law, Chapters 1, 5, 9):

  Lab 18 — Drive-Thru (Series queues / Tandem):
    3 stations in series: Order → Pay → Pickup
    Bottleneck = station with highest utilization
    Overall throughput limited by bottleneck rate

  Lab 19 — ER Simulation:
    Patients classified: Critical, Urgent, Non-Urgent
    Priority queuing: Critical served first (LCFS/Priority)
    Multiple resources: triage nurses, doctors, beds
    Track: wait by priority, bed utilization, throughput

  Lab 20 — Statistical Output Analysis (Law, Ch 9):
    Given: Y = [3.2, 4.3, 5.1, 4.2, 4.6] (5 replications)
    Point estimate: Ȳ = mean(Y)
    Variance: S² = Σ(Yi-Ȳ)²/(n-1)
    Standard Error: SE = S/√n
    95% CI: Ȳ ± t_{n-1, 0.025} × SE
    Required replications for precision ε:
      n* = ⌈(t_{n-1,0.025} × S / ε)²⌉
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from tabulate import tabulate
import math


# ═══════════════════════════════════════════════════════════
# LAB 18 — FAST-FOOD DRIVE-THRU
# ═══════════════════════════════════════════════════════════

def simulate_drive_thru(lam, mu_order, mu_pay, mu_pickup,
                         duration, seed=42):
    """
    3-stage tandem queue: Order → Pay → Pickup (all single-server).
    Entity moves to next stage only when current server is free.
    """
    rng = np.random.default_rng(seed)

    stages = {
        'order' : {'mu': mu_order,  'busy': False, 'depart': float('inf'),
                   'queue': [], 'total_wait': 0, 'served': 0,
                   'busy_area': 0.0},
        'pay'   : {'mu': mu_pay,    'busy': False, 'depart': float('inf'),
                   'queue': [], 'total_wait': 0, 'served': 0,
                   'busy_area': 0.0},
        'pickup': {'mu': mu_pickup, 'busy': False, 'depart': float('inf'),
                   'queue': [], 'total_wait': 0, 'served': 0,
                   'busy_area': 0.0},
    }
    stage_names = ['order', 'pay', 'pickup']

    arrival_t   = -1/lam * np.log(rng.random())
    clock       = 0.0
    last_t      = 0.0
    total_served = 0
    customer_id  = 0
    total_sojourn = 0.0

    # Each customer: {id, stage, arrive_stage_time}
    in_system = {}

    while clock < duration:
        # Build event list
        events = [('arrival', arrival_t, None)]
        for sn, st in stages.items():
            if st['depart'] < float('inf'):
                events.append(('depart_' + sn, st['depart'], sn))
        events.sort(key=lambda x: x[1])
        etype, t_next, sname = events[0]

        if t_next > duration:
            break

        # Accumulate busy areas
        dt = t_next - last_t
        for st in stages.values():
            if st['busy']:
                st['busy_area'] += dt
        last_t = t_next
        clock  = t_next

        if etype == 'arrival':
            customer_id += 1
            st = stages['order']
            if not st['busy']:
                st['busy']    = True
                st['depart']  = clock + (-1/st['mu'] * np.log(rng.random()))
                st['total_wait'] += 0
            else:
                st['queue'].append((clock, customer_id))
            arrival_t = clock + (-1/lam * np.log(rng.random()))
            in_system[customer_id] = clock

        elif etype.startswith('depart_'):
            sn = sname
            st = stages[sn]
            st['served'] += 1

            # Move to next stage
            si = stage_names.index(sn)
            if si < len(stage_names) - 1:
                next_sn = stage_names[si + 1]
                nst = stages[next_sn]
                wait = clock
                if not nst['busy']:
                    nst['busy']   = True
                    nst['depart'] = clock + (-1/nst['mu'] * np.log(rng.random()))
                    nst['total_wait'] += 0
                else:
                    nst['queue'].append((clock, -1))
            else:
                # Customer leaves system
                cid = None  # simplified
                total_served += 1

            # Next customer in current stage's queue
            if st['queue']:
                arr_t, _ = st['queue'].pop(0)
                wait      = clock - arr_t
                st['total_wait'] += wait
                st['depart']  = clock + (-1/st['mu'] * np.log(rng.random()))
            else:
                st['busy']   = False
                st['depart'] = float('inf')

    utilization = {sn: st['busy_area'] / duration for sn, st in stages.items()}
    avg_wait    = {sn: (st['total_wait'] / max(st['served'], 1))
                   for sn, st in stages.items()}
    throughput  = min(st['served'] for st in stages.values()) / duration

    return utilization, avg_wait, stages, throughput


def lab18_drive_thru():
    print("\n" + "="*65)
    print("   LAB 18: FAST-FOOD DRIVE-THRU BOTTLENECK ANALYSIS")
    print("="*65)

    try:
        lam       = float(input("\n  Arrival rate λ (cars/min)         [e.g. 0.8]: "))
        mu_order  = float(input("  Order station service rate μ₁     [e.g. 1.2]: "))
        mu_pay    = float(input("  Pay station service rate μ₂        [e.g. 2.0]: "))
        mu_pickup = float(input("  Pickup station service rate μ₃     [e.g. 1.0]: "))
        duration  = float(input("  Simulation duration (min)          [e.g.1000]: "))
        seed      = int(input("  Random seed                        [e.g.  42]: "))
    except ValueError:
        lam, mu_order, mu_pay, mu_pickup = 0.8, 1.2, 2.0, 1.0
        duration, seed = 1000, 42

    util, avg_wait, stages, throughput = simulate_drive_thru(
        lam, mu_order, mu_pay, mu_pickup, duration, seed)

    stage_names = ['order', 'pay', 'pickup']
    stage_rates = {'order': mu_order, 'pay': mu_pay, 'pickup': mu_pickup}

    # Theoretical utilizations (ρ = λ/μ for tandem)
    rho_th = {sn: lam / stage_rates[sn] for sn in stage_names}

    bottleneck = max(util, key=util.get)

    print(f"\n  ── Stage Performance ──")
    rows = []
    for sn in stage_names:
        rows.append([
            sn.capitalize(),
            f"μ={stage_rates[sn]}/min",
            f"{stages[sn]['served']}",
            f"{util[sn]:.4f} ({util[sn]*100:.1f}%)",
            f"{rho_th[sn]:.4f} ({rho_th[sn]*100:.1f}%)",
            f"{avg_wait[sn]:.4f} min",
            "⚠ BOTTLENECK" if sn == bottleneck else "OK",
        ])
    print(tabulate(rows,
                   headers=["Stage", "Rate", "Served", "Sim Util", "Theo Util",
                             "Avg Wait", "Status"],
                   tablefmt="rounded_grid"))

    print(f"\n  ── System Summary ──")
    print(tabulate([
        ["Bottleneck Stage",       bottleneck.capitalize()],
        ["Bottleneck Utilization", f"{util[bottleneck]*100:.1f}%"],
        ["System Throughput",      f"{throughput:.4f} cars/min"],
        ["Max Possible (theory)",  f"{min(stage_rates.values()):.4f} cars/min"],
    ], headers=["Metric", "Value"], tablefmt="rounded_grid"))

    # ── PLOTS ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lab 18 — Drive-Thru Bottleneck Analysis",
                 fontsize=13, fontweight='bold', color='#2c3e50')

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bn_clr = ['#e74c3c' if sn == bottleneck else '#3498db'
              for sn in stage_names]

    # A: Utilization comparison
    ax1 = axes[0]
    x  = np.arange(3)
    w  = 0.35
    b1 = ax1.bar(x - w/2, [util[s]*100 for s in stage_names], w,
                  color=bn_clr, alpha=0.85, label='Simulated')
    b2 = ax1.bar(x + w/2, [rho_th[s]*100 for s in stage_names], w,
                  color='#bdc3c7', alpha=0.85, label='Theoretical ρ')
    ax1.axhline(100, color='red', linestyle='--', linewidth=1.5, label='100%')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Order', 'Pay', 'Pickup'])
    ax1.set_title('Station Utilization (%)\n(Red = Bottleneck)', fontweight='bold')
    ax1.set_ylabel('Utilization (%)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('#f8f9fa')

    # B: Avg waiting time per stage
    ax2 = axes[1]
    wts = [avg_wait[s] for s in stage_names]
    bars = ax2.bar(['Order', 'Pay', 'Pickup'], wts,
                   color=bn_clr, alpha=0.85, edgecolor='white')
    ax2.bar_label(bars, fmt='%.3f', fontsize=9)
    ax2.set_title('Avg Waiting Time per Stage', fontweight='bold')
    ax2.set_ylabel('Avg Wait (min)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#f8f9fa')

    # C: System diagram
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 4)
    ax3.axis('off')
    ax3.set_title('Drive-Thru Layout', fontweight='bold')
    box_x = [1, 4, 7]
    box_lbl = ['ORDER\n(μ₁)', 'PAY\n(μ₂)', 'PICKUP\n(μ₃)']
    for i, (bx, lbl) in enumerate(zip(box_x, box_lbl)):
        clr = '#e74c3c' if stage_names[i] == bottleneck else '#3498db'
        rect = plt.Rectangle((bx, 1.2), 2, 1.6, fill=True,
                              facecolor=clr, alpha=0.7, edgecolor='white',
                              linewidth=2, zorder=2)
        ax3.add_patch(rect)
        ax3.text(bx+1, 2.0, lbl, ha='center', va='center',
                 fontsize=10, fontweight='bold', color='white', zorder=3)
        ax3.text(bx+1, 1.0, f"ρ={util[stage_names[i]]*100:.0f}%",
                 ha='center', fontsize=9, color='#2c3e50', zorder=3)
        if i < 2:
            ax3.annotate('', xy=(box_x[i+1], 2.0), xytext=(bx+2, 2.0),
                         arrowprops=dict(arrowstyle='->', color='#2c3e50',
                                        lw=2), zorder=4)
    ax3.text(0.2, 2.0, '→ Cars', fontsize=10, va='center', color='#2c3e50')
    ax3.text(9.2, 2.0, '→', fontsize=12, va='center', color='#2c3e50')
    red_patch = mpatches.Patch(color='#e74c3c', label='Bottleneck')
    blue_patch = mpatches.Patch(color='#3498db', label='Normal')
    ax3.legend(handles=[red_patch, blue_patch], loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab18_drive_thru.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab18_drive_thru.png")
    plt.show()
    print("\n✅ Lab 18 Complete!\n")


# ═══════════════════════════════════════════════════════════
# LAB 19 — EMERGENCY ROOM (ER) PATIENT FLOW
# ═══════════════════════════════════════════════════════════

def lab19_er_simulation():
    print("\n" + "="*65)
    print("   LAB 19: EMERGENCY ROOM PATIENT FLOW SIMULATION")
    print("="*65)

    try:
        lam_c  = float(input("\n  Critical arrival rate (pts/hr)    [e.g. 2.0]: "))
        lam_u  = float(input("  Urgent arrival rate (pts/hr)      [e.g. 5.0]: "))
        lam_n  = float(input("  Non-urgent arrival rate (pts/hr)  [e.g. 8.0]: "))
        n_doc  = int(input("  Number of doctors                 [e.g.   3]: "))
        mu_tr  = float(input("  Triage rate μ_triage (pts/hr)     [e.g.  12]: "))
        mu_tx  = float(input("  Treatment rate μ_treat (pts/hr)   [e.g.   4]: "))
        dur    = float(input("  Simulation duration (hrs)         [e.g.  24]: "))
        seed   = int(input("  Random seed                       [e.g.  42]: "))
    except ValueError:
        lam_c, lam_u, lam_n = 2.0, 5.0, 8.0
        n_doc, mu_tr, mu_tx = 3, 12.0, 4.0
        dur, seed = 24, 42

    rng   = np.random.default_rng(seed)
    clock = 0.0

    # Priority: 0=Critical, 1=Urgent, 2=Non-Urgent
    priority_names = ['Critical', 'Urgent', 'Non-Urgent']
    lam_list       = [lam_c, lam_u, lam_n]

    # State
    triage_busy  = False
    triage_dep   = float('inf')
    triage_queue = []   # (priority, arrival_time, patient_id)

    doc_busy  = [False] * n_doc
    doc_dep   = [float('inf')] * n_doc
    doc_queue = []  # (priority, entry_time, patient_id)

    total_wait_triage = [0.0, 0.0, 0.0]
    total_wait_doc    = [0.0, 0.0, 0.0]
    n_triage_served   = [0, 0, 0]
    n_doc_served      = [0, 0, 0]
    total_sojourn     = [0.0, 0.0, 0.0]
    triage_busy_t     = 0.0
    doc_busy_t        = [0.0] * n_doc
    last_t            = 0.0

    # Schedule first arrivals
    next_arr = [clock + (-1/lam * np.log(rng.random())) for lam in lam_list]
    patient_id = 0
    patient_arrive = {}  # id → system arrival time

    while clock < dur:
        # Build events
        ev = [(t, 'arr', i) for i, t in enumerate(next_arr)]
        if triage_dep < float('inf'):
            ev.append((triage_dep, 'triage_dep', -1))
        for i, d in enumerate(doc_dep):
            if d < float('inf'):
                ev.append((d, 'doc_dep', i))
        ev.sort()
        t_next, etype, idx = ev[0]

        if t_next > dur:
            break

        dt = t_next - last_t
        if triage_busy:
            triage_busy_t += dt
        for i in range(n_doc):
            if doc_busy[i]:
                doc_busy_t[i] += dt
        last_t = t_next
        clock  = t_next

        if etype == 'arr':
            priority   = idx
            patient_id += 1
            patient_arrive[patient_id] = clock
            # Join triage queue
            triage_queue.append((priority, clock, patient_id))
            triage_queue.sort(key=lambda x: x[0])  # priority sort
            # If triage free
            if not triage_busy:
                p, arr_t, pid = triage_queue.pop(0)
                triage_busy = True
                wait        = clock - arr_t
                total_wait_triage[p] += wait
                triage_dep = clock + (-1/mu_tr * np.log(rng.random()))
                triage_queue_pid = (p, pid)
            next_arr[priority] = clock + (-1/lam_list[priority] * np.log(rng.random()))

        elif etype == 'triage_dep':
            # Find what was just triaged
            cur = getattr(lab19_er_simulation, '_cur_triage', None)
            if cur:
                p, pid = cur
            else:
                p, pid = 0, patient_id
            n_triage_served[p] += 1

            # Send to doctor queue
            doc_queue.append((p, clock, pid))
            doc_queue.sort(key=lambda x: x[0])

            # Assign to free doctor
            free_doc = next((i for i in range(n_doc) if not doc_busy[i]), None)
            if free_doc is not None and doc_queue:
                dp, de_t, dpid = doc_queue.pop(0)
                doc_busy[free_doc] = True
                doc_dep[free_doc]  = clock + (-1/mu_tx * np.log(rng.random()))
                total_wait_doc[dp] += (clock - de_t)

            # Start next triage
            if triage_queue:
                p2, arr_t2, pid2 = triage_queue.pop(0)
                lab19_er_simulation._cur_triage = (p2, pid2)
                triage_busy = True
                wait2       = clock - arr_t2
                total_wait_triage[p2] += wait2
                triage_dep  = clock + (-1/mu_tr * np.log(rng.random()))
            else:
                triage_busy = False
                triage_dep  = float('inf')
                lab19_er_simulation._cur_triage = None

        elif etype == 'doc_dep':
            i = idx
            doc_busy[i] = False
            doc_dep[i]  = float('inf')
            n_doc_served[idx % 3] += 1

            if doc_queue:
                dp, de_t, dpid = doc_queue.pop(0)
                doc_busy[i] = True
                wait        = clock - de_t
                total_wait_doc[dp] += wait
                n_doc_served[dp]   += 0
                doc_dep[i] = clock + (-1/mu_tx * np.log(rng.random()))

    # ── Results ──
    lam_total = sum(lam_list)
    print(f"\n  ── ER Performance Metrics ──")
    print(tabulate([
        ["Total arrival rate",   f"{lam_total:.1f} pts/hr"],
        ["Triage utilization",   f"{triage_busy_t/dur*100:.1f}%"],
        ["Doctor utilization",   f"{np.mean(doc_busy_t)/dur*100:.1f}% (avg)"],
    ], headers=["Metric", "Value"], tablefmt="rounded_grid"))

    print("\n  ── Wait Times by Priority ──")
    wt_rows = []
    for p, pname in enumerate(priority_names):
        avg_triage_w = total_wait_triage[p] / max(n_triage_served[p], 1)
        avg_doc_w    = total_wait_doc[p]    / max(n_doc_served[p], 1)
        wt_rows.append([
            pname,
            f"{n_triage_served[p]}",
            f"{avg_triage_w*60:.1f} min",
            f"{avg_doc_w*60:.1f} min",
            f"{(avg_triage_w + avg_doc_w)*60:.1f} min",
        ])
    print(tabulate(wt_rows,
                   headers=["Priority", "Patients", "Triage Wait",
                             "Doctor Wait", "Total Wait"],
                   tablefmt="rounded_grid"))

    # ── PLOTS ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lab 19 — Emergency Room Patient Flow Simulation",
                 fontsize=13, fontweight='bold', color='#2c3e50')
    clrs  = ['#e74c3c', '#f39c12', '#3498db']

    # A: Patient volume
    ax1 = axes[0]
    volumes = [n_triage_served[p] for p in range(3)]
    ax1.bar(priority_names, volumes, color=clrs, alpha=0.85, edgecolor='white')
    for i, v in enumerate(volumes):
        ax1.text(i, v + 0.3, str(v), ha='center', fontweight='bold')
    ax1.set_title('Patients Served by Priority', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('#f8f9fa')

    # B: Wait times
    ax2 = axes[1]
    tw_min = [(total_wait_triage[p]/max(n_triage_served[p],1) +
               total_wait_doc[p]/max(n_doc_served[p],1)) * 60
              for p in range(3)]
    bars = ax2.bar(priority_names, tw_min, color=clrs, alpha=0.85, edgecolor='white')
    ax2.bar_label(bars, fmt='%.1f min', fontsize=9)
    ax2.set_title('Avg Total Wait by Priority', fontweight='bold')
    ax2.set_ylabel('Total Wait (min)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#f8f9fa')

    # C: ER resource utilization
    ax3 = axes[2]
    resources   = ['Triage'] + [f'Doc {i+1}' for i in range(n_doc)]
    utils_pct   = [triage_busy_t/dur*100] + [d/dur*100 for d in doc_busy_t]
    r_colors    = ['#9b59b6'] + ['#27ae60']*n_doc
    bars2 = ax3.bar(resources, utils_pct, color=r_colors, alpha=0.85,
                    edgecolor='white')
    ax3.bar_label(bars2, fmt='%.1f%%', fontsize=8)
    ax3.axhline(100, color='red', linestyle='--', linewidth=1.5)
    ax3.set_title('Resource Utilization (%)', fontweight='bold')
    ax3.set_ylabel('Utilization (%)')
    ax3.set_ylim(0, 115)
    ax3.tick_params(axis='x', rotation=30)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab19_er_simulation.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab19_er_simulation.png")
    plt.show()
    print("\n✅ Lab 19 Complete!\n")


# ═══════════════════════════════════════════════════════════
# LAB 20 — STATISTICAL OUTPUT ANALYSIS (PHARMACY)
# ═══════════════════════════════════════════════════════════

def lab20_output_analysis():
    print("\n" + "="*65)
    print("   LAB 20: STATISTICAL OUTPUT ANALYSIS — PHARMACY")
    print("="*65)
    print("\n  Given data: 5 replication waiting times (minutes)")
    print("  Y = [3.2, 4.3, 5.1, 4.2, 4.6]")

    try:
        raw = input("\n  Enter replication values (comma-separated)\n"
                    "  [Press Enter for default: 3.2,4.3,5.1,4.2,4.6]: ").strip()
        if raw:
            Y = [float(x.strip()) for x in raw.split(',')]
        else:
            Y = [3.2, 4.3, 5.1, 4.2, 4.6]
        eps   = float(input("  Desired half-width precision ε [e.g. 0.5]: "))
        alpha = float(input("  Significance level α          [e.g. 0.05]: "))
    except (ValueError, EOFError):
        Y, eps, alpha = [3.2, 4.3, 5.1, 4.2, 4.6], 0.5, 0.05

    n     = len(Y)
    Y_bar = np.mean(Y)
    S2    = np.var(Y, ddof=1)
    S     = np.sqrt(S2)
    SE    = S / np.sqrt(n)

    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    hw     = t_crit * SE
    CI_lo  = Y_bar - hw
    CI_hi  = Y_bar + hw

    R_star = math.ceil((t_crit * S / eps) ** 2)

    print(f"\n  ── Data Summary ──")
    rep_rows = [(i+1, y) for i, y in enumerate(Y)]
    print(tabulate(rep_rows, headers=["Replication i", "Wq_i (min)"],
                   tablefmt="rounded_grid"))

    print(f"\n  ── Step-by-Step Computation ──")
    comp_rows = [
        ["n (replications)",                 str(n)],
        ["Ȳ = Σ Yᵢ/n   (Point Estimate)",   f"{Y_bar:.4f} min"],
        ["S² = Σ(Yᵢ-Ȳ)²/(n-1)  (Variance)", f"{S2:.4f}"],
        ["S = √S²   (Std Dev)",               f"{S:.4f}"],
        ["SE = S/√n  (Std Error)",            f"{SE:.4f}"],
        ["t_{n-1, α/2} = t_{%d, %.3f}" % (n-1, alpha/2), f"{t_crit:.4f}"],
        ["Half-Width h = t × SE",             f"{hw:.4f} min"],
        ["─"*38,                              "─"*20],
        [f"95% CI = Ȳ ± h",                  f"[{CI_lo:.4f},  {CI_hi:.4f}]"],
        ["─"*38,                              "─"*20],
        [f"Required reps for ε={eps} min",   f"n* = {R_star} replications"],
    ]
    print(tabulate(comp_rows, headers=["Step", "Value"],
                   tablefmt="rounded_grid"))

    # Show how n* was derived
    print(f"\n  ── Formula for Required Replications ──")
    print(f"  n* = ⌈(t_{{n-1,α/2}} × S / ε)²⌉")
    print(f"     = ⌈({t_crit:.4f} × {S:.4f} / {eps})²⌉")
    inner = t_crit * S / eps
    print(f"     = ⌈{inner:.4f}²⌉")
    print(f"     = ⌈{inner**2:.4f}⌉")
    print(f"     = {R_star}  ← You need {R_star} replications for ε={eps} min precision")

    # ── PLOTS ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Lab 20 — Statistical Output Analysis: Drive-Thru Pharmacy",
        fontsize=12, fontweight='bold', color='#2c3e50')

    reps = list(range(1, n+1))

    # A: Individual observations + CI
    ax1 = axes[0]
    ax1.bar(reps, Y, color='#3498db', alpha=0.8, edgecolor='white')
    ax1.axhline(Y_bar, color='#e74c3c', linestyle='-',
                linewidth=2.5, label=f'Ȳ = {Y_bar:.3f}')
    ax1.axhline(CI_lo, color='#e74c3c', linestyle='--',
                linewidth=1.5, label=f'95% CI')
    ax1.axhline(CI_hi, color='#e74c3c', linestyle='--', linewidth=1.5)
    ax1.fill_between([0.5, n+0.5], CI_lo, CI_hi,
                     alpha=0.12, color='#e74c3c')
    for i, y in enumerate(Y):
        ax1.text(i+1, y+0.05, f'{y}', ha='center', fontsize=10,
                 fontweight='bold')
    ax1.set_title('Replication Data + 95% CI', fontweight='bold')
    ax1.set_xlabel('Replication')
    ax1.set_ylabel('Avg Waiting Time (min)')
    ax1.set_xticks(reps)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_facecolor('#f8f9fa')

    # B: t-distribution + CI visualization
    ax2 = axes[1]
    df_val  = n - 1
    x_t     = np.linspace(-5, 5, 400)
    pdf_t   = stats.t.pdf(x_t, df=df_val)
    t_obs   = (Y_bar - Y_bar) / SE  # = 0 (centered)
    ax2.plot(x_t, pdf_t, '#2c3e50', linewidth=2.5,
             label=f't-dist (df={df_val})')
    ax2.fill_between(x_t, pdf_t, where=x_t <= -t_crit,
                     color='#e74c3c', alpha=0.4, label='Rejection region')
    ax2.fill_between(x_t, pdf_t, where=x_t >= t_crit,
                     color='#e74c3c', alpha=0.4)
    ax2.axvline(-t_crit, color='red', linestyle='--', linewidth=2)
    ax2.axvline( t_crit, color='red', linestyle='--', linewidth=2,
                label=f'±t_crit = ±{t_crit:.3f}')
    ax2.set_title(f't-Distribution (df={df_val})\nCritical region α={alpha}',
                  fontweight='bold')
    ax2.set_xlabel('t')
    ax2.set_ylabel('Density')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # C: Half-width vs n (required replications)
    ax3 = axes[2]
    n_vals  = np.arange(2, 50)
    hw_vals = []
    for nv in n_vals:
        tc  = stats.t.ppf(1 - alpha/2, df=nv-1)
        hw_vals.append(tc * S / np.sqrt(nv))
    ax3.plot(n_vals, hw_vals, color='#3498db', linewidth=2.5)
    ax3.axhline(eps, color='red', linestyle='--',
                linewidth=2, label=f'ε = {eps}')
    ax3.axvline(R_star, color='green', linestyle=':',
                linewidth=2, label=f'n* = {R_star}')
    ax3.scatter([n], [hw], color='orange', s=120, zorder=5,
                label=f'Current n={n}, h={hw:.3f}')
    ax3.set_title('Half-Width vs Replications', fontweight='bold')
    ax3.set_xlabel('Number of Replications n')
    ax3.set_ylabel('Half-Width h (min)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab20_output_analysis.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab20_output_analysis.png")
    plt.show()
    print("\n✅ Lab 20 Complete!\n")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║  CIT 324 — Lab 18, 19, 20                       ║")
    print("╚══════════════════════════════════════════════════╝")
    print("\n  [18] Lab 18 — Drive-Thru Bottleneck")
    print("  [19] Lab 19 — ER Patient Flow")
    print("  [20] Lab 20 — Statistical Output Analysis")
    print("  [0]  Run ALL")

    try:
        ch = int(input("\n  Choice [0/18/19/20]: "))
    except ValueError:
        ch = 0

    if ch == 18 or ch == 0:
        lab18_drive_thru()
    if ch == 19 or ch == 0:
        lab19_er_simulation()
    if ch == 20 or ch == 0:
        lab20_output_analysis()


if __name__ == "__main__":
    main()
