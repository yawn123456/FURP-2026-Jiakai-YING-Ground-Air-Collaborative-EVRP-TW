"""
Visualization utilities for HMOA results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from model import Solution


def plot_pareto_front(pareto_front: List[Solution],
                      title: str = "HMOA Pareto Front",
                      save_path: str = None,
                      highlight_compromise: bool = True):
    """
    Plot the Pareto front of solutions.

    Args:
        pareto_front: List of non-dominated solutions
        title: Plot title
        save_path: Path to save the figure (None = show)
        highlight_compromise: Whether to highlight best-compromise solution
    """
    if not pareto_front:
        print("No solutions to plot.")
        return

    # Sort by cost
    pareto_front.sort(key=lambda s: s.cost)

    costs = [s.cost for s in pareto_front]
    sats = [s.satisfaction for s in pareto_front]

    plt.figure(figsize=(10, 6))

    # Plot Pareto front
    plt.plot(costs, sats, 'b.-', markersize=8, linewidth=1.5, label='Pareto Front')
    plt.scatter(costs, sats, c='blue', s=30, zorder=5)

    # Highlight best-compromise solution
    if highlight_compromise and len(pareto_front) > 1:
        min_cost = min(costs)
        max_sat = max(sats)
        best_dist = float('inf')
        best_idx = 0
        for i, (c, s) in enumerate(zip(costs, sats)):
            norm_c = (c - min_cost) / (max_sat - min_cost + 1) if max_sat > min_cost else 0
            norm_s = (max_sat - s) / (max_sat + 1)
            dist = np.sqrt(norm_c**2 + norm_s**2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        plt.scatter(costs[best_idx], sats[best_idx], c='red', s=120,
                   marker='*', zorder=10, label='Best Compromise')

    # Highlight extreme points
    if len(pareto_front) > 1:
        min_cost_idx = np.argmin(costs)
        max_sat_idx = np.argmax(sats)
        plt.scatter(costs[min_cost_idx], sats[min_cost_idx], c='green', s=80,
                   marker='s', zorder=8, label='Best Cost')
        plt.scatter(costs[max_sat_idx], sats[max_sat_idx], c='orange', s=80,
                   marker='^', zorder=8, label='Best Satisfaction')

    plt.xlabel('f1: Transportation Cost', fontsize=12)
    plt.ylabel('f2: Customer Satisfaction', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_convergence(history: dict, save_path: str = None):
    """
    Plot convergence metrics over generations.
    """
    gens = history.get('generations', [])
    best_costs = history.get('best_costs', [])
    best_sats = history.get('best_sats', [])
    pf_sizes = history.get('pf_sizes', [])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if best_costs:
        axes[0].plot(gens, best_costs, 'b-', linewidth=1.5)
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Best Cost')
        axes[0].set_title('Cost Convergence')
        axes[0].grid(True, alpha=0.3)

    if best_sats:
        axes[1].plot(gens, best_sats, 'g-', linewidth=1.5)
        axes[1].set_xlabel('Generation')
        axes[1].set_ylabel('Best Satisfaction')
        axes[1].set_title('Satisfaction Convergence')
        axes[1].grid(True, alpha=0.3)

    if pf_sizes:
        axes[2].plot(gens, pf_sizes, 'r-', linewidth=1.5)
        axes[2].set_xlabel('Generation')
        axes[2].set_ylabel('PF Size')
        axes[2].set_title('Pareto Front Growth')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
