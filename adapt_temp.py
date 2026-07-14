# ============================================================
# PRL-Ready High-Resolution Figure Export
# Combined Recentering + Adaptive Temperature Cure
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ------------------ PRL-Style Matplotlib Settings ------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "patch.linewidth": 0.8,
})

np.random.seed(42)

# ------------------ Parameters ------------------
n = 8
T_base = np.sqrt(n)
lr = 0.05
n_steps = 300
switch_step = 80

rx = np.random.uniform(0.5, 0.75, n)
theta = np.random.randn(n) * 0.3

# ------------------ Master formula ------------------
def G_T(E, sigma, T):
    if sigma < 1e-8:
        return (1.0 / np.cosh(E / T))**2
    def integrand(z):
        arg = (E + sigma * z) / T
        return (1.0 / np.cosh(arg))**2 * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    res, _ = quad(integrand, -8, 8, epsabs=1e-7, limit=100)
    return res

def compute_Ebar_sigma_local(theta, rx):
    Ebar = np.dot(theta, rx)
    vars_j = 1.0 - rx**2
    sigmaQ = np.sqrt(np.dot(theta**2, vars_j))
    return Ebar, sigmaQ, rx.copy()

def master_grads_combined(theta, rx, T_base, recenter=False, adaptive_T=False):
    Ebar, sigmaQ, local_exps = compute_Ebar_sigma_local(theta, rx)
    if recenter:
        E_eff = 0.0
        T_eff = max(abs(Ebar), T_base) if adaptive_T else T_base
        g = G_T(E_eff, sigmaQ, T_eff)
    else:
        T_eff = T_base
        g = G_T(Ebar, sigmaQ, T_base)
    grads = (local_exps / T_eff) * g
    return grads, Ebar, sigmaQ, g, T_eff

def run_combined_cure(switch_step=80):
    theta_t = theta.copy()
    history = {'step': [], 'Ebar': [], 'grad_norm': [], 'G': [], 'T_eff': []}

    for step in range(n_steps):
        recenter_now = step >= switch_step
        adaptive_now = step >= switch_step
        grads, Ebar, sigmaQ, g, T_eff = master_grads_combined(
            theta_t, rx, T_base, recenter=recenter_now, adaptive_T=adaptive_now
        )
        theta_t += lr * grads
        history['step'].append(step)
        history['Ebar'].append(Ebar)
        history['grad_norm'].append(np.linalg.norm(grads))
        history['G'].append(g)
        history['T_eff'].append(T_eff)
    return history

# ------------------ Run simulations ------------------
hist = run_combined_cure(switch_step=switch_step)
hist_no = run_combined_cure(switch_step=n_steps + 1)

# ------------------ Create Figure ------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))  # PRL-friendly size (double column)

fig.suptitle(
    r"Training Dynamics — Quantized Neuron ($n=8$)" "\n"
    r"Combined Cure: Recentering + Adaptive Temperature (activated at step 80)",
    fontsize=11, fontweight='bold'
)

# Panel (a) - Energy drift
ax = axes[0, 0]
ax.plot(hist_no['step'], hist_no['Ebar'], label="No cure", color="#d62728", lw=1.8, alpha=0.75)
ax.plot(hist['step'], hist['Ebar'], label="With combined cure", color="#2ca02c", lw=2.0)
ax.axvline(switch_step, color="#9467bd", ls="--", lw=1.5, alpha=0.9)
ax.axhline(T_base, color="gray", ls=":", lw=1.0, alpha=0.7, label=r"$T \approx \sqrt{n}$")
ax.set_xlabel("GD step")
ax.set_ylabel(r"$\bar{E}$")
ax.set_title("(a) Energy drift")
ax.legend(loc="upper left", frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.5)

# Panel (b) - Gradient magnitude
ax = axes[0, 1]
ax.semilogy(hist_no['step'], hist_no['grad_norm'], label="No cure", color="#d62728", lw=1.8, alpha=0.75)
ax.semilogy(hist['step'], hist['grad_norm'], label="With combined cure", color="#2ca02c", lw=2.0)
ax.axvline(switch_step, color="#9467bd", ls="--", lw=1.5, alpha=0.9)
ax.set_xlabel("GD step")
ax.set_ylabel(r"$\|\nabla_\theta f\|$ (typical)")
ax.set_title("(b) Gradient magnitude — plateau then recovery")
ax.legend(loc="upper right", frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.5)

# Panel (c) - Gaussian-dressed gain
ax = axes[1, 0]
ax.plot(hist_no['step'], hist_no['G'], label="No cure", color="#d62728", lw=1.8, alpha=0.75)
ax.plot(hist['step'], hist['G'], label="With combined cure", color="#2ca02c", lw=2.0)
ax.axvline(switch_step, color="#9467bd", ls="--", lw=1.5, alpha=0.9)
ax.set_xlabel("GD step")
ax.set_ylabel(r"$\mathcal{G}_T(\bar{E}, \sigma_Q)$")
ax.set_title("(c) Gaussian-dressed gain (master formula)")
ax.legend(loc="upper right", frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.5)

# Panel (d) - Adaptive temperature
ax = axes[1, 1]
ax.plot(hist['step'], hist['T_eff'], color="#9467bd", lw=2.0, label=r"$T_{\rm eff}$ (adaptive)")
ax.axvline(switch_step, color="#9467bd", ls="--", lw=1.5, alpha=0.9)
ax.axhline(T_base, color="gray", ls=":", lw=1.0, alpha=0.7)
ax.set_xlabel("GD step")
ax.set_ylabel(r"$T_{\rm eff}$")
ax.set_title("(d) Adaptive temperature after activation")
ax.legend(loc="upper left", frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# ------------------ PRL-Ready Export ------------------
plt.savefig(
    "fig4_combined_cure.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.08,
    dpi=300,
    transparent=False
)

# Optional: also save a high-resolution PNG for preview
plt.savefig(
    "fig4_combined_cure.png",
    format="png",
    bbox_inches="tight",
    pad_inches=0.08,
    dpi=300
)

print("Figure saved successfully:")
print("  → fig4_combined_cure.pdf  (vector, PRL-ready)")
print("  → fig4_combined_cure.png  (high-res preview)")
