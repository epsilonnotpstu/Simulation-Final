"""
=============================================================
CIT 324 - Simulation and Modeling Sessional
Lab 02: Inventory Management Simulation
=============================================================
Theory (Averill M. Law, Chapter 1 & Appendix 1B):
  - Inventory simulation tracks stock over time
  - Demand arrives randomly (Poisson process here)
  - When stock hits reorder point → place order
  - Order arrives after lead time (Uniform random)
  - Costs: Holding cost (per unit/day) + Shortage cost
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tabulate import tabulate

def inventory_simulation(
    initial_stock=100,
    reorder_point=20,
    order_qty=50,
    mean_demand=10.0,
    lead_time_min=2,
    lead_time_max=5,
    holding_cost=2.0,
    shortage_cost=10.0,
    order_cost=50.0,
    sim_days=365,
    seed=42
):
    """
    Simple (s, Q) inventory simulation.
      s = reorder_point
      Q = order_qty
    """
    rng = np.random.default_rng(seed)

    stock           = initial_stock
    on_order        = False          # Is an order pending?
    order_due_day   = None           # When will order arrive?

    total_holding   = 0.0
    total_shortage  = 0.0
    total_orders    = 0
    total_order_cost= 0.0
    days_stockout   = 0

    # ── History for plotting ──
    history_stock   = []
    history_day     = []
    history_demand  = []
    history_order   = []
    order_events    = []   # (day_placed, day_arrived)

    print("\n" + "="*60)
    print("   INVENTORY MANAGEMENT SIMULATION")
    print("="*60)
    print(f"   Initial Stock       : {initial_stock} units")
    print(f"   Reorder Point (s)   : {reorder_point} units")
    print(f"   Order Quantity (Q)  : {order_qty} units")
    print(f"   Mean Daily Demand   : {mean_demand} units/day")
    print(f"   Lead Time           : [{lead_time_min}, {lead_time_max}] days")
    print(f"   Holding Cost        : ${holding_cost}/unit/day")
    print(f"   Shortage Cost       : ${shortage_cost}/unit/day")
    print(f"   Fixed Order Cost    : ${order_cost}/order")
    print(f"   Simulation Period   : {sim_days} days")
    print("="*60)

    for day in range(1, sim_days + 1):

        # ── Check if pending order arrives today ──
        if on_order and day >= order_due_day:
            stock    += order_qty
            on_order  = False
            order_events[-1] = (order_events[-1][0], day)

        # ── Generate today's demand (Poisson) ──
        demand = rng.poisson(mean_demand)

        # ── Fulfill demand ──
        if demand <= stock:
            stock -= demand
            shortage_today = 0
        else:
            shortage_today = demand - stock
            stock = 0
            days_stockout += 1

        # ── Accumulate costs ──
        total_holding  += stock * holding_cost
        total_shortage += shortage_today * shortage_cost

        # ── Check reorder ──
        placed_today = False
        if stock <= reorder_point and not on_order:
            on_order       = True
            total_orders  += 1
            total_order_cost += order_cost
            lead           = rng.integers(lead_time_min, lead_time_max + 1)
            order_due_day  = day + lead
            order_events.append((day, None))
            placed_today   = True

        history_stock.append(stock)
        history_day.append(day)
        history_demand.append(demand)
        history_order.append(placed_today)

    total_cost = total_holding + total_shortage + total_order_cost

    results = {
        'total_holding'   : total_holding,
        'total_shortage'  : total_shortage,
        'total_order_cost': total_order_cost,
        'total_cost'      : total_cost,
        'total_orders'    : total_orders,
        'days_stockout'   : days_stockout,
        'avg_stock'       : np.mean(history_stock),
        'min_stock'       : np.min(history_stock),
        'max_stock'       : np.max(history_stock),
        'history_stock'   : history_stock,
        'history_day'     : history_day,
        'history_demand'  : history_demand,
        'history_order'   : history_order,
        'order_events'    : order_events,
        'sim_days'        : sim_days,
    }
    return results


def print_results(res):
    print("\n" + "─"*60)
    print("   SIMULATION RESULTS SUMMARY")
    print("─"*60)

    rows = [
        ["Total Holding Cost",    f"${res['total_holding']:.2f}"],
        ["Total Shortage Cost",   f"${res['total_shortage']:.2f}"],
        ["Total Ordering Cost",   f"${res['total_order_cost']:.2f}"],
        ["─"*30,                  "─"*15],
        ["TOTAL COST",            f"${res['total_cost']:.2f}"],
        ["─"*30,                  "─"*15],
        ["Number of Orders",      f"{res['total_orders']}"],
        ["Days with Stockout",    f"{res['days_stockout']} days"],
        ["Average Stock Level",   f"{res['avg_stock']:.1f} units"],
        ["Minimum Stock Reached", f"{res['min_stock']} units"],
        ["Maximum Stock Level",   f"{res['max_stock']} units"],
    ]
    print(tabulate(rows, headers=["Metric", "Value"],
                   tablefmt="rounded_grid"))


def plot_results(res):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle("Inventory Management Simulation — Dashboard",
                 fontsize=14, fontweight='bold', color='#2c3e50')

    days   = res['history_day']
    stocks = res['history_stock']

    # ── Plot 1: Stock level over time ──
    ax1 = axes[0]
    ax1.plot(days, stocks, color='#2980b9', linewidth=1.2, label='Stock Level')
    ax1.axhline(y=20, color='#e74c3c', linestyle='--',
                linewidth=1.5, label=f'Reorder Point (s=20)')
    ax1.axhline(y=np.mean(stocks), color='#27ae60', linestyle=':',
                linewidth=1.5, label=f'Avg Stock = {np.mean(stocks):.1f}')
    ax1.fill_between(days, stocks, alpha=0.15, color='#2980b9')
    # Mark order placements
    order_days = [d for d, placed in zip(days, res['history_order']) if placed]
    ax1.scatter(order_days, [0]*len(order_days),
                color='#e67e22', zorder=5, s=40, marker='^',
                label='Order Placed')
    ax1.set_ylabel('Stock Level (units)', fontsize=10)
    ax1.set_title('Inventory Level Over Time', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    # ── Plot 2: Daily Demand ──
    ax2 = axes[1]
    ax2.bar(days, res['history_demand'],
            color='#8e44ad', alpha=0.7, width=1.0, label='Daily Demand')
    ax2.axhline(y=np.mean(res['history_demand']), color='#e74c3c',
                linestyle='--', linewidth=1.5,
                label=f"Mean Demand = {np.mean(res['history_demand']):.1f}")
    ax2.set_ylabel('Units Demanded', fontsize=10)
    ax2.set_title('Daily Demand (Poisson Process)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_facecolor('#f8f9fa')

    # ── Plot 3: Cost Breakdown (pie) ──
    ax3 = axes[2]
    costs  = [res['total_holding'], res['total_shortage'], res['total_order_cost']]
    labels = ['Holding Cost', 'Shortage Cost', 'Ordering Cost']
    clrs   = ['#3498db', '#e74c3c', '#2ecc71']
    wedges, texts, autotexts = ax3.pie(
        costs, labels=labels, colors=clrs,
        autopct='%1.1f%%', startangle=140,
        textprops={'fontsize': 10})
    ax3.set_title(f'Cost Breakdown  (Total = ${res["total_cost"]:.2f})',
                  fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lab02_inventory.png',
                dpi=150, bbox_inches='tight')
    print("\n✅ Plot saved: lab02_inventory.png")
    plt.show()


def main():
    print("\n╔══════════════════════════════════════════╗")
    print("║  CIT 324 — Lab 02: Inventory Simulation  ║")
    print("╚══════════════════════════════════════════╝")

    try:
        init   = int(input("\n  Initial Stock Level      [e.g. 100]: "))
        rop    = int(input("  Reorder Point (s)        [e.g.  20]: "))
        oq     = int(input("  Order Quantity (Q)       [e.g.  50]: "))
        demand = float(input("  Mean Daily Demand        [e.g.  10]: "))
        days   = int(input("  Simulation Days          [e.g. 365]: "))
        hc     = float(input("  Holding Cost ($/unit/day)[e.g. 2.0]: "))
        sc     = float(input("  Shortage Cost ($/unit)   [e.g.10.0]: "))
    except ValueError:
        print("  ⚠ Using defaults.")
        init, rop, oq, demand, days, hc, sc = 100, 20, 50, 10.0, 365, 2.0, 10.0

    res = inventory_simulation(
        initial_stock=init, reorder_point=rop,
        order_qty=oq, mean_demand=demand,
        sim_days=days, holding_cost=hc,
        shortage_cost=sc
    )
    print_results(res)
    plot_results(res)
    print("\n✅ Lab 02 Complete!\n")


if __name__ == "__main__":
    main()
