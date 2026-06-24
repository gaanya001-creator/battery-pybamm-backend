# server.py — FastAPI + PyBaMM backend for Electrochemical Cell Simulator v11
# Compatible with PyBaMM >= 23.x
#
# LOCAL SETUP:
#   pip install "pybamm>=23.0" fastapi uvicorn numpy
#   uvicorn server:app --reload --host 0.0.0.0 --port 8000
#
# RENDER DEPLOY:
#   1. Push this file + requirements.txt to a GitHub repo
#   2. New Web Service on render.com → connect repo
#   3. Build Command : pip install -r requirements.txt
#   4. Start Command : uvicorn server:app --host 0.0.0.0 --port $PORT
#   5. Instance type : Free (or Starter for no cold-start)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import pybamm
import time
import traceback
import numpy as np

app = FastAPI(title="PyBaMM Battery API", version="1.1.0")

# Allow all origins so the HTML file can call this from any host
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── constants ────────────────────────────────────────────────────────────────
VALID_MODELS   = {"DFN", "SPMe", "SPM", "MPM"}
VALID_PARAMS   = {"Chen2020", "Marquis2019", "OKane2022", "Ecker2015"}
VALID_THERMALS = {"isothermal", "lumped", "x-lumped"}

# PyBaMM renamed the variable between versions — we try newest name first
VOLTAGE_KEYS = ("Voltage [V]", "Terminal voltage [V]")

# All known temperature variable names across PyBaMM versions
TEMP_KEYS = (
    "Volume-averaged cell temperature [K]",
    "Cell temperature [K]",
    "X-averaged cell temperature [K]",
)

# Electrolyte concentration keys (mol/m³ → divide by 1000 for mol/L)
CE_KEYS = (
    ("Electrolyte concentration [mol.m-3]", 1 / 1000),
    ("Electrolyte concentration [Molar]",   1.0),
)

# ── request schema ────────────────────────────────────────────────────────────
class SimRequest(BaseModel):
    model:               str   = "DFN"
    parameter_set:       str   = "Chen2020"
    c_rate:              float = 1.0
    temperature_celsius: float = 25.0
    geometry:            str   = "pouch"
    thermal_model:       str   = "lumped"
    initial_soc:         float = 0.5

    @validator("c_rate")
    def c_rate_range(cls, v):
        if not 0.01 <= v <= 10:
            raise ValueError("c_rate must be between 0.01 and 10")
        return v

    @validator("initial_soc")
    def soc_range(cls, v):
        if not 0.0 < v <= 1.0:
            raise ValueError("initial_soc must be in (0, 1]")
        return v

    @validator("temperature_celsius")
    def temp_range(cls, v):
        if not -40 <= v <= 80:
            raise ValueError("temperature_celsius must be between -40 and 80")
        return v

# ── helpers ───────────────────────────────────────────────────────────────────
def safe_last(sol, key) -> float | None:
    """Return the last scalar value of a solution variable, or None if missing."""
    try:
        v = sol[key].entries
        return float(v.flat[-1])
    except Exception:
        return None


def try_keys(sol, key_list):
    """Try each key in key_list; return the entries array of the first match."""
    for key in key_list:
        try:
            return sol[key].entries
        except Exception:
            continue
    return None


def to_list_1d(arr, t_idx=-1):
    """Convert a (nx,) or (nx, nt) array to a flat Python list at time index t_idx."""
    if arr is None:
        return []
    arr = np.array(arr)
    if arr.ndim == 2:
        arr = arr[:, t_idx]
    return arr.tolist()

# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":         "ok",
        "pybamm_version": pybamm.__version__,
        "solver":         "CasADi/IDAKLU",
    }


@app.post("/simulate")
def simulate(req: SimRequest):
    t0 = time.time()
    try:
        # ── validation ──────────────────────────────────────────────────────
        if req.model not in VALID_MODELS:
            raise HTTPException(400, f"Unknown model '{req.model}'. Valid: {VALID_MODELS}")
        if req.parameter_set not in VALID_PARAMS:
            raise HTTPException(400, f"Unknown param set '{req.parameter_set}'. Valid: {VALID_PARAMS}")

        # ── build model options ──────────────────────────────────────────────
        thermal = req.thermal_model if req.thermal_model in VALID_THERMALS else "lumped"
        options = {"thermal": thermal}

        # FIX 1: PyBaMM's "cell geometry" only accepts 'pouch' or 'arbitrary'.
        # UI dropdowns expose 'cylindrical'/'coin' which are not valid PyBaMM
        # values — passing them straight through caused an OptionError (500).
        if req.model == "DFN":
            options["cell geometry"] = "pouch" if req.geometry == "pouch" else "arbitrary"

        # ── instantiate model ────────────────────────────────────────────────
        if req.model == "MPM":
            model = pybamm.lithium_ion.MPM(options=options)
        else:
            ModelClass = getattr(pybamm.lithium_ion, req.model)
            model = ModelClass(options=options)

        # ── parameter set ────────────────────────────────────────────────────
        param = pybamm.ParameterValues(req.parameter_set)
        T_ref = req.temperature_celsius + 273.15
        param["Ambient temperature [K]"] = T_ref
        param.update({"Initial temperature [K]": T_ref}, check_already_exists=False)

        # ── experiment ───────────────────────────────────────────────────────
        experiment = pybamm.Experiment([
            pybamm.step.string(f"Discharge at {req.c_rate}C until 2.5 V")
        ])
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
        sim.solve(initial_soc=req.initial_soc)
        sol = sim.solution

        # ── voltage ──────────────────────────────────────────────────────────
        # FIX 2: PyBaMM renamed "Terminal voltage [V]" → "Voltage [V]" in v23.9.
        # We try newest name first so this works on all versions.
        V_entries = try_keys(sol, VOLTAGE_KEYS)
        if V_entries is None:
            raise HTTPException(
                500,
                "Could not find voltage variable in solution. "
                f"Tried: {VOLTAGE_KEYS}. "
                f"Available variables (sample): {list(sol.all_models[0].variables.keys())[:20]}"
            )
        V_arr = V_entries.tolist()
        V_end  = float(V_arr[-1])

        # ── time + capacity ──────────────────────────────────────────────────
        t_arr    = sol["Time [h]"].entries.tolist()
        cap_ah   = safe_last(sol, "Discharge capacity [A.h]") or 0.0
        cap_mAh  = cap_ah * 1000

        # ── temperature ──────────────────────────────────────────────────────
        T_arr = [T_ref] * len(t_arr)
        if thermal != "isothermal":
            T_entries = try_keys(sol, TEMP_KEYS)
            if T_entries is not None:
                arr = np.array(T_entries)
                T_arr = (arr if arr.ndim == 1 else arr[0]).tolist()
        T_max = float(np.max(T_arr))

        # ── electrolyte concentration (spatial profile at t_end) ─────────────
        ce_prof = []
        for key, factor in CE_KEYS:
            try:
                arr = np.array(sol[key].entries)
                col = arr[:, -1] if arr.ndim == 2 else arr
                ce_prof = (col * factor).tolist()
                break
            except Exception:
                continue

        # ── solid concentration (stoichiometry) ──────────────────────────────
        cs_n, cs_p = [], []
        for side, cmax_key, out in [
            ("negative", "Maximum concentration in negative electrode [mol.m-3]", cs_n),
            ("positive", "Maximum concentration in positive electrode [mol.m-3]", cs_p),
        ]:
            try:
                cmax = float(param[cmax_key])
                key  = f"X-averaged {side} particle surface concentration [mol.m-3]"
                val  = float(np.array(sol[key].entries).flat[-1]) / cmax
                out.extend([max(0.0, min(1.0, val))] * 15)
            except Exception:
                pass

        # ── SEI thickness ────────────────────────────────────────────────────
        sei_nm = 0.0
        for key in (
            "Total SEI thickness [m]",
            "X-averaged total SEI thickness [m]",
            "Negative electrode SEI film thickness [m]",
        ):
            v = safe_last(sol, key)
            if v is not None:
                sei_nm = v * 1e9
                break

        # ── overpotential ────────────────────────────────────────────────────
        eta_mv = 0.0
        for key in (
            "Negative electrode reaction overpotential [V]",
            "X-averaged negative electrode reaction overpotential [V]",
        ):
            v = safe_last(sol, key)
            if v is not None:
                eta_mv = abs(v) * 1000
                break

        # ── energy density (rough: mAh × |V_end| / 100) ─────────────────────
        energy_density = cap_mAh * abs(V_end) / 100.0

        return {
            # time-series
            "V_hist":               V_arr,
            "T_hist":               T_arr,
            "t_hist":               t_arr,
            # spatial profiles
            "ce_profile":           ce_prof,
            "cs_n_profile":         cs_n,
            "cs_p_profile":         cs_p,
            # scalars
            "V_end":                V_end,
            "T_max":                T_max,
            "capacity_mAh":         cap_mAh,
            "ce_min":               min(ce_prof) if ce_prof else 0.8,
            "ce_max":               max(ce_prof) if ce_prof else 1.2,
            "sei_thickness_nm":     sei_nm,
            "energy_density_Wh_kg": energy_density,
            "overpotential_mV":     eta_mv,
            "solve_time_ms":        round((time.time() - t0) * 1000, 1),
            "model":                req.model,
            "params":               req.parameter_set,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            500,
            detail=str(e) + "\n" + traceback.format_exc()[-1200:]
        )
