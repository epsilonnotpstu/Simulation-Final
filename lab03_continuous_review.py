"""
=============================================================
CIT 324 - Simulation and Modeling Sessional
Lab 03: Inventory System Simulation (Continuous Review)
=============================================================
Theory (Averill M. Law, Appendix 1B — Inventory example):
  - Demand ~ Normal(μ, σ²) — more realistic than Poisson
  - (s, S) policy: reorder point s, order-up-to level S
    → when stock falls to s, order (S - current_stock) units
  - Holding cost: h × avg inventory
  - Shortage cost: p × units short
  - Track over one fiscal year (252 working days)
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import stats


def continuous_review_simulation(
    initial_stock    = 200,
    reorder_point_s  = 50,
    order_up_to_S    = 200,
    demand_mean      = 20.0,
    demand_std       = 5.0,
    lead_time_mean   = 3.0,
    lead_time_std    = 1.0,
    holding_cost     = 1.5,
    shortage_cost    = 15.0,
    order_cost       = 100.0,
    fiscal_days      = 252,
    seed             = 42
):
    """
    (s, S) Continuous Review Inventory Simulation.
    Demand: Normal(demand_mean, demand_std)
    Lead Time: Normal(lead_time_mean, lead_time_std) clipped ≥ 1
    """
    rng   = np.random.default_rng(seed)
    stock = initial_stock

    # Pending orders: list of (arrival_day)
    pending_orders = []

    total_holding  = 0.0
    total_shortage = 0.0
    total_ord_cost = 0.0
    total_orders   = 0
    days_stockout  = 0
    total_unmet    = 0

    hist_stock    = [stock]
    hist_demand   = []
    hist_order    = []
    hist_shortage = []

    print("\n" + "="*62)
    print("   CONTINUOUS REVIEW INVENTORY (s, S) SIMULATION")
    print("="*62)
    print(f"   Initial Stock       : {initial_stock} units")
    print(f"   Reorder Point  (s)  : {reorder_point_s} units")
    print(f"   Order-Up-To    (S)  : {order_up_to_S} units")
    print(f"   Demand              : Normal(μ={demand_mean}, σ={demand_std})")
    print(f"   Lead Time           : Normal(μ={lead_time_mean}, σ={lead_time_std})")
    print(f"   Holding Cost        : ${holding_cost}/unit/day")
    print(f"   Shortage Cost       : ${shortage_cost}/unit/day")
    print(f"   Fiscal Year         : {fiscal_days} working days")
    print("="*62)

    for day in range(1, fiscal_days + 1):

        # ── Receive any orders due today ──
        arrived = [d for d in pending_orders if d <= day]
        for d in arrived:
            qty    = order_up_to_S - stock   # dynamic order qty
            stock += max(qty, 0)
            pending_orders.remove(d)

        # ── Generate demand (Normal, clipped ≥ 0) ──
        demand = max(0, round(rng.normal(demand_mean, demand_std)))

        # ── Fulfill demand ──
        if demand <= stock:
            stock         -= demand
            shortage_today = 0
        else:
            shortage_today = demand - stock
            stock          = 0
            days_stockout += 1
            total_unmet   += shortage_today

        # ── Accumulate costs ──
        total_holding  += stock * holding_cost
        total_shortage += shortage_today * shortage_cost

        placed = False

        # ── Continuous review: check immediately after each demand ──
        if stock <= reorder_point_s and len(pending_orders) == 0:
            lead    = max(1, round(rng.normal(lead_time_mean, lead_time_std)))
            pending_orders.append(day + lead)
            total_ord_cost += order_cost
            total_orders   += 1
            placed          = True

        hist_stock.append(stock)
        hist_demand.append(demand)
        hist_order.append(placed)
        hist_shortage.append(shortage_today)

    total_cost = total_holding + total_shortage + total_ord_cost

    return {
        'total_holding' : total_holding,
        'total_shortage': total_shortage,
        'total_ord_cost': total_ord_cost,
        'total_cost'    : total_cost,
        'total_orders'  : total_orders,
        'days_stockout' : days_stockout,
        'total_unmet'   : total_unmet,
        'avg_stock'     : np.mean(hist_stock),
        'hist_stock'    : hist_stock,
        'hist_demand'   : hist_demand,
        'hist_order'    : hist_order,
        'hist_shortage' : hist_shortage,
        'fiscal_days'   : fiscal_days,
        'reorder_point' : reorder_point_s,
        'order_up_to'   : order_up_to_S,
    }


def print_results(res):
    print("\n" + "─"*62)
    print("   RESULTS — (s,S) Continuous Review Inventory")
    print("─"*62)
    service_level = 100 * (1 - res['days_stockout'] / res['fiscal_days'])
    rows = [
        ["Total Holding Cost",       f"${res['total_holding']:.2f}"],
        ["Total Shortage Cost",      f"${res['total_shortage']:.2f}"],
        ["Total Ordering Cost",      f"${res['total_ord_cost']:.2f}"],
        ["─"*32,                     "─"*15],
        ["TOTAL ANNUAL COST",        f"${res['total_cost']:.2f}"],
        ["─"*32,                     "─"*15],
        ["Number of Orders Placed",  f"{res['total_orders']}"],
        ["Days with Stockout",       f"{res['days_stockout']}"],
        ["Total Unmet Demand",       f"{res['total_unmet']} units"],
        ["Average Stock Level",      f"{res['avg_stock']:.1f} units"],
        ["Service Level",            f"{service_level:.1f}%"],
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="rounded_grid"))


def plot_results(res):
    days = list(range(res['fiscal_days'] + 1))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle("Continuous Review Inventory (s,S) — Fiscal Year Dashboard",
                 fontsize=14, fontweight='bold', color='#2c3e50')

    # ── Plot 1: Stock level ──
    ax1 = axes[0]
    ax1.plot(days, res['hist_stock'], color='#2980b9', linewidth=1.2)
    ax1.axhline(res['reorder_point'], color='#e74c3c', linestyle='--',
                linewidth=1.5, label=f"Reorder Point s={res['reorder_point']}")
    ax1.axhline(res['order_up_to'], color='#27ae60', linestyle=':',
                linewidth=1.5, label=f"Order-Up-To S={res['order_up_to']}")
    ax1.axhline(res['avg_stock'], color='#8e44ad', linestyle='-.',
                linewidth=1.3, label=f"Avg Stock={res['avg_stock']:.1f}")
    ax1.fill_between(days, res['hist_stock'], alpha=0.15, color='#2980b9')
    order_x = [d+1 for d, placed in enumerate(res['hist_order']) if placed]
    if order_x:
        ax1.scatter(order_x, [5]*len(order_x),
                    marker='^', color='#e67e22', s=50, zorder=5,
                    label='Order Placed')
    ax1.set_ylabel('Stock (units)')
    ax1.set_title('Inventory Level — Continuous Review (s,S) Policy',
                  fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # ── Plot 2: Demand histogram with fitted normal ──
    ax2 = axes[1]
    demands = res['hist_demand']
    ax2.hist(demands, bins=25, color='#8e44ad', alpha=0.7,
             density=True, label='Simulated Demand')
    xr = np.linspace(min(demands), max(demands), 200)
    mu_, std_ = np.mean(demands), np.std(demands)
    ax2.plot(xr, stats.norm.pdf(xr, mu_, std_),
             'r-', linewidth=2.5,
             label=f'Normal fit: μ={mu_:.1f}, σ={std_:.1f}')
    ax2.set_xlabel('Daily Demand (units)')
    ax2.set_ylabel('Density')
    ax2.set_title('Daily Demand Distribution (Normal)', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')

    # ── Plot 3: Cumulative cost ──
    ax3 = axes[2]
    cum_h = np.cumsum([s * 1.5 for s in res['hist_stock'][1:]])
    cum_s = np.cumsum([sh * 15 for sh in res['hist_shortage']])
    cum_d = list(range(1, res['fiscal_days'] + 1))
    ax3.plot(cum_d, cum_h, color='#3498db', linewidth=2, label='Cumulative Holding Cost')
    ax3.plot(cum_d, cum_s, color='#e74c3c', linewidth=2, label='Cumulative Shortage Cost')
    ax3.plot(cum_d, cum_h + cum_s, color='#2ecc71', linewidth=2.5,
             linestyle='--', label='Total (H + S)')
    ax3.set_xlabel('Working Day')
    ax3.set_ylabel('Cumulative Cost ($)')
    ax3.set_title('Cumulative Cost Over Fiscal Year', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab03_continuous_review.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab03_continuous_review.png")
    plt.show()


def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║  CIT 324 — Lab 03: Continuous Review (s,S)  ║")
    print("╚══════════════════════════════════════════════╝")

    try:
        s      = int(input("\n  Reorder Point  (s)         [e.g.  50]: "))
        S      = int(input("  Order-Up-To Level (S)      [e.g. 200]: "))
        d_mu   = float(input("  Mean Daily Demand          [e.g.  20]: "))
        d_std  = float(input("  Std Dev of Demand          [e.g.   5]: "))
        lt_mu  = float(input("  Mean Lead Time (days)      [e.g.   3]: "))
        hc     = float(input("  Holding Cost ($/unit/day)  [e.g. 1.5]: "))
        sc     = float(input("  Shortage Cost ($/unit/day) [e.g.  15]: "))
        days   = int(input("  Fiscal Year Working Days   [e.g. 252]: "))
    except ValueError:
        print("  ⚠ Using defaults.")
        s, S, d_mu, d_std, lt_mu = 50, 200, 20.0, 5.0, 3.0
        hc, sc, days = 1.5, 15.0, 252

    res = continuous_review_simulation(
        reorder_point_s=s, order_up_to_S=S,
        demand_mean=d_mu, demand_std=d_std,
        lead_time_mean=lt_mu,
        holding_cost=hc, shortage_cost=sc,
        fiscal_days=days
    )
    print_results(res)
    plot_results(res)
    print("\n✅ Lab 03 Complete!\n")


if __name__ == "__main__":
    main()
