import pybamm
import numpy as np
import csv
import matplotlib.pyplot as plt

# --- model setup ---
model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
param = pybamm.ParameterValues("Chen2020")
param["Ambient temperature [K]"] = 298.15
param.update({"Initial temperature [K]": 298.15}, check_already_exists=False)

experiment = pybamm.Experiment([
    pybamm.step.string("Discharge at 1C until 2.5 V")
])

sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
sim.solve(initial_soc=0.9)
sol = sim.solution

# --- extract results ---
t = sol["Time [h]"].entries

# voltage (try both names for compatibility)
V = None
for key in ("Voltage [V]", "Terminal voltage [V]"):
    try:
        V = sol[key].entries
        break
    except Exception:
        pass

# temperature
T = None
for key in ("Volume-averaged cell temperature [K]",
            "Cell temperature [K]",
            "X-averaged cell temperature [K]"):
    try:
        T = sol[key].entries
        break
    except Exception:
        pass
if T is None:
    T = np.full_like(t, 298.15)

# --- save CSV ---
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time [h]", "Voltage [V]", "Temperature [K]"])
    for row in zip(t, V, T):
        writer.writerow([round(x, 6) for x in row])
print("Saved: results.csv")

# --- save plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(t, V, color="#1e88e5")
ax1.set_ylabel("Voltage [V]")
ax1.set_title("DFN Model — 1C Discharge (Chen2020)")
ax1.grid(True, alpha=0.3)

ax2.plot(t, T, color="#e53935")
ax2.set_ylabel("Temperature [K]")
ax2.set_xlabel("Time [h]")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("Saved: results.png")
print(f"\nFinal voltage : {V[-1]:.4f} V")
print(f"Max temperature: {max(T):.2f} K")
