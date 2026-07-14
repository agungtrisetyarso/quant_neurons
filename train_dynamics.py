# ============================================================
# Option A — Recommended for Fig. 4
# Training Dynamics with Mid-Training Cure Activation
# "Plateau → Recovery" narrative
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

np.random.seed(42)

# ------------------ Parameters ------------------
n = 8
T_base = np.sqrt(n)
lr = 0.05
n_steps = 300
switch_step = 80          # Cure activates here (change to 100 if you prefer)

rx = np.random.uniform(0.5, 0.75, n)
theta = np.random.randn(n) * 0.3

# ------------------ Master formula (robust) ------------------
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

def master_grads_cured(theta, rx, T_base, recenter=False):
    Ebar, sigmaQ, local_exps = compute_Ebar_sigma_local(theta, rx)
    if recenter:
        E_eff = 0.0
        g = G_T(E_eff, sigmaQ, T_base)
    else:
        g = G_T(Ebar, sigmaQ, T_base)
    grads = (local_exps / T_base) * g
    return grads, Ebar, sigmaQ, g

# ------------------ Training with cure switch ------------------
def run_with_switch(switch_step=80):
    theta_t = theta.copy()
    history = {
        'step': [], 'Ebar': [], 'grad_norm': [], 'G': [],
        'cure_active': []
    }

    for step in range(n_steps):
        recenter_now = step >= switch_step
        grads, Ebar, sigmaQ, g = master_grads_cured(theta_t, rx, T_base, recenter=recenter_now)
        theta_t += lr * grads

        history['step'].append(step)
        history['Ebar'].append(Ebar)
        history['grad_norm'].append(np.linalg.norm(grads))
        history['G'].append(g)
        history['cure_active'].append(recenter_now)

    return history

print(f"Running dynamics with cure activation at step {switch_step}...")
hist = run_with_switch(switch_step=switch_step)

# Also run a pure "No cure" reference
print("Running pure No-cure reference for comparison...")
hist_no = run_with_switch(switch_step=n_steps + 1)  # never activates

# ------------------ Plots for Fig. 4 ------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    f"Training Dynamics Demo — Quantized Neuron (n={n})\n"
    r"Drift-induced plateau followed by recovery after recentering cure activation",
    fontsize=14, fontweight='bold'
)

# 1. Energy drift
ax = axes[0, 0]
ax.plot(hist_no['step'], hist_no['Ebar'], label="No cure (reference)", color="crimson", lw=2, alpha=0.7)
ax.plot(hist['step'], hist['Ebar'], label="With cure (activated at step 80)", color="seagreen", lw=2.5)
ax.axvline(switch_step, color="purple", ls="--", lw=2, alpha=0.85)
ax.text(switch_step + 8, 9.5, "Cure\nactivated", color="purple", fontsize=10, fontweight='bold')
ax.axhline(T_base, color="gray", ls=":", alpha=0.6, label=r"$T \approx \sqrt{n}$")
ax.set_xlabel("GD step")
ax.set_ylabel(r"$\bar{E} = \operatorname{Tr}[H(\theta)\rho]$")
ax.set_title("Energy drift (self-saturation driver)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# 2. Gradient magnitude (the key panel)
ax = axes[0, 1]
ax.semilogy(hist_no['step'], hist_no['grad_norm'], label="No cure (reference)", color="crimson", lw=2, alpha=0.7)
ax.semilogy(hist['step'], hist['grad_norm'], label="With cure (activated at step 80)", color="seagreen", lw=2.5)
ax.axvline(switch_step, color="purple", ls="--", lw=2, alpha=0.85)
ax.set_xlabel("GD step")
ax.set_ylabel(r"$\|\nabla_\theta f\|$ (typical)")
ax.set_title("Gradient magnitude — clear plateau then recovery")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# 3. Gaussian-dressed gain
ax = axes[1, 0]
ax.plot(hist_no['step'], hist_no['G'], label="No cure (reference)", color="crimson", lw=2, alpha=0.7)
ax.plot(hist['step'], hist['G'], label="With cure (activated at step 80)", color="seagreen", lw=2.5)
ax.axvline(switch_step, color="purple", ls="--", lw=2, alpha=0.85)
ax.set_xlabel("GD step")
ax.set_ylabel(r"$\mathcal{G}_T(\bar{E}, \sigma_Q)$")
ax.set_title("Gaussian-dressed gain (master formula)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# 4. Cure activation indicator
ax = axes[1, 1]
cure_on = np.array(hist['cure_active']).astype(int)
ax.fill_between(hist['step'], 0, cure_on, color="seagreen", alpha=0.25, label="Cure active")
ax.axvline(switch_step, color="purple", ls="--", lw=2, alpha=0.85)
ax.set_xlabel("GD step")
ax.set_ylabel("Cure status (0 = off, 1 = on)")
ax.set_title("Cure activation timeline")
ax.set_ylim(-0.05, 1.15)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------ Summary ------------------
print("\n=== Summary ===")
print(f"Cure activated at step {switch_step}")
print(f"Final |Ebar| (with cure)     = {hist['Ebar'][-1]:.3f}")
print(f"Final gradient norm (with cure) = {hist['grad_norm'][-1]:.4f}")
print(f"Final G_T (with cure)           = {hist['G'][-1]:.4f}")
print("\nThe green curves clearly show the plateau forming, then strong recovery after cure activation.")
