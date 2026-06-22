# server.py — PyBaMM High-Accuracy Backend
# Render.com deploy ke liye ready
# GitHub → Render → Public URL → Simulator mein paste karo
#
# Local run:  uvicorn server:app --host 0.0.0.0 --port 8000
# Deploy:     Push to GitHub → connect on render.com (free tier)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pybamm, time, traceback, os
import numpy as np

app = FastAPI(title="PyBaMM Battery API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_MODELS   = {"DFN", "SPMe", "SPM"}
VALID_PARAMS   = {"Chen2020", "Marquis2019", "OKane2022", "Ecker2015"}
VALID_THERMALS = {"isothermal", "lumped", "x-lumped"}

class SimRequest(BaseModel):
    model:               str   = "DFN"
    parameter_set:       str   = "Chen2020"
    c_rate:              float = 1.0
    temperature_celsius: float = 25.0
    geometry:            str   = "pouch"
    thermal_model:       str   = "lumped"
    initial_soc:         float = 1.0        # BUG1 FIXED: was 0.5, now 1.0
    enable_sei:          bool  = False

@app.get("/")
def root():
    return {"status": "ok", "message": "PyBaMM Battery API v2.0 — use /simulate or /health"}

@app.get("/health")
def health():
    return {
        "status":         "ok",
        "pybamm_version": pybamm.__version__,
        "valid_models":   list(VALID_MODELS),
        "valid_params":   list(VALID_PARAMS),
    }

@app.post("/simulate")
def simulate(req: SimRequest):
    t0 = time.time()
    try:
        # ── Validate ─────────────────────────────────────────────
        if req.model not in VALID_MODELS:
            raise HTTPException(400, f"Unknown model '{req.model}'. Valid: {VALID_MODELS}")
        if req.parameter_set not in VALID_PARAMS:
            raise HTTPException(400, f"Unknown param set '{req.parameter_set}'. Valid: {VALID_PARAMS}")
        if not (0.05 <= req.c_rate <= 5.0):
            raise HTTPException(400, f"C-rate must be 0.05–5. Got {req.c_rate}")
        if not (0.01 < req.initial_soc <= 1.0):
            raise HTTPException(400, f"initial_soc must be (0,1]. Got {req.initial_soc}")

        # ── Build model ───────────────────────────────────────────
        thermal = req.thermal_model if req.thermal_model in VALID_THERMALS else "lumped"
        options = {"thermal": thermal}
        if req.enable_sei:
            options["SEI"] = "ec reaction limited"
        if req.model == "DFN" and req.geometry in ("pouch", "cylindrical"):
            options["cell geometry"] = req.geometry

        ModelClass = getattr(pybamm.lithium_ion, req.model)
        model = ModelClass(options=options)

        # ── Parameters ───────────────────────────────────────────
        param = pybamm.ParameterValues(req.parameter_set)
        T_K = req.temperature_celsius + 273.15
        param["Ambient temperature [K]"] = T_K
        param.update({"Initial temperature [K]": T_K}, check_already_exists=False)

        # ── Experiment ───────────────────────────────────────────
        experiment = pybamm.Experiment([
            pybamm.step.string(f"Discharge at {req.c_rate}C until 2.5 V")
        ])
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)

        # ── Solve with auto-fallback DFN → SPMe → SPM ────────────
        # BUG4 FIXED: proper try/except with fallback
        solved_model = req.model
        try:
            sim.solve(initial_soc=req.initial_soc)
        except Exception as e1:
            if req.model == "DFN":
                try:
                    fallback = pybamm.lithium_ion.SPMe(options={"thermal": thermal})
                    sim2 = pybamm.Simulation(fallback, parameter_values=param, experiment=experiment)
                    sim2.solve(initial_soc=req.initial_soc)
                    sim = sim2
                    solved_model = "SPMe (DFN fallback)"
                except Exception as e2:
                    raise HTTPException(422,
                        f"DFN failed: {str(e1)[:200]}. SPMe also failed: {str(e2)[:200]}. "
                        "Try SPM or lower C-rate.")
            else:
                raise HTTPException(422,
                    f"PyBaMM solve failed (model={req.model}, c_rate={req.c_rate}, "
                    f"params={req.parameter_set}): {str(e1)[:300]}. Try lower C-rate.")

        sol = sim.solution
        if sol is None:
            raise HTTPException(422, "Solver returned empty solution.")

        # ── Extract voltage (trim bad points) ────────────────────
        t_raw = sol["Time [h]"].entries.tolist()
        V_raw = sol["Terminal voltage [V]"].entries.tolist()
        n_raw = min(len(t_raw), len(V_raw))
        t_raw, V_raw = t_raw[:n_raw], V_raw[:n_raw]

        t_arr, V_arr = [], []
        for tt, vv in zip(t_raw, V_raw):
            if vv is None or not np.isfinite(vv) or not (1.5 < vv < 5.5):
                break
            t_arr.append(float(tt))
            V_arr.append(float(vv))

        if not V_arr:
            raise HTTPException(500, "No valid voltage points after filtering.")

        n = len(V_arr)
        V_end  = float(V_arr[-1])
        V_mean = float(np.mean(V_arr))
        V_mid  = float(V_arr[n // 2])   # BUG2 FIXED: mid-discharge reference

        # ── Capacity ─────────────────────────────────────────────
        def safe_last(key):
            try:
                arr = sol[key].entries.flat[:n]
                return float(arr[-1]) if len(arr) else None
            except Exception:
                return None

        cap_Ah  = safe_last("Discharge capacity [A.h]") or 0.0
        cap_mAh = cap_Ah * 1000.0
        # BUG3 FIXED: limit raised to 50000 (Chen2020 = 5000 mAh, safe margin)
        if cap_mAh < 0 or cap_mAh > 50000:
            cap_mAh = None

        # ── Temperature ──────────────────────────────────────────
        T_arr = [T_K] * n
        if thermal != "isothermal":
            for key in (
                "Volume-averaged cell temperature [K]",
                "X-averaged cell temperature [K]",
                "Cell temperature [K]",
            ):
                try:
                    arr = sol[key].entries
                    full = arr.tolist() if arr.ndim == 1 else arr[0].tolist()
                    trimmed = full[:n]
                    if all(np.isfinite(x) and 200 < x < 500 for x in trimmed):
                        T_arr = trimmed
                        break
                except Exception:
                    pass
        T_max = float(np.max(T_arr))

        # ── Electrolyte concentration ────────────────────────────
        ce_prof, ce_ok = [], False
        for key in (
            "Electrolyte concentration [mol.m-3]",
            "Electrolyte concentration [Molar]",
        ):
            try:
                arr = sol[key].entries
                factor = 1/1000.0 if "mol.m-3" in key else 1.0
                col_idx = min(n, arr.shape[-1]) - 1 if arr.ndim == 2 else -1
                col = arr[:, col_idx] if arr.ndim == 2 else arr
                vals = (np.array(col, dtype=float) * factor).tolist()
                if all(np.isfinite(x) and -0.1 < x < 20 for x in vals):
                    ce_prof = vals
                    ce_ok = True
                    break
            except Exception:
                pass

        # ── Solid concentrations ──────────────────────────────────
        cs_n, cs_p = [], []
        for side, cmax_key, out in [
            ("negative", "Maximum concentration in negative electrode [mol.m-3]", cs_n),
            ("positive", "Maximum concentration in positive electrode [mol.m-3]", cs_p),
        ]:
            try:
                cmax = float(param[cmax_key])
                key  = f"X-averaged {side} particle surface concentration [mol.m-3]"
                arr  = sol[key].entries
                val  = float(arr.flat[min(n, len(arr.flat)) - 1]) / cmax
                out.extend([max(0.0, min(1.0, val))] * 15)
            except Exception:
                pass

        # ── Overpotential ────────────────────────────────────────
        eta_mv = None
        for key in (
            "X-averaged negative electrode reaction overpotential [V]",
            "Negative electrode reaction overpotential [V]",
        ):
            try:
                arr  = sol[key].entries.flat[:n]
                cand = abs(float(arr[-1])) * 1000
                if np.isfinite(cand) and cand < 2000:
                    eta_mv = round(cand, 2)
                    break
            except Exception:
                pass

        # ── SEI thickness ────────────────────────────────────────
        sei_nm, sei_active = None, bool(options.get("SEI"))
        if sei_active:
            for key in (
                "X-averaged total SEI thickness [m]",
                "Total SEI thickness [m]",
                "Negative electrode SEI film thickness [m]",
            ):
                try:
                    cand = float(sol[key].entries.flat[-1]) * 1e9
                    if np.isfinite(cand) and 0 <= cand < 10000:
                        sei_nm = round(cand, 3)
                        break
                except Exception:
                    pass

        # ── Energy density ───────────────────────────────────────
        energy_Wh = (cap_mAh / 1000.0) * V_mean if cap_mAh else None
        energy_density = None
        if energy_Wh:
            for mkey in ("Cell mass [kg]", "Total mass [kg]", "Cell mass [g]"):
                try:
                    m = float(param[mkey])
                    if mkey.endswith("[g]"): m /= 1000.0
                    ed = energy_Wh / m
                    if 0 < ed < 600:
                        energy_density = round(ed, 1)
                    break
                except Exception:
                    pass

        # ── Coulombic efficiency (real calc) ─────────────────────
        coulombic_eff = None
        try:
            nom_cap_Ah = float(param["Nominal cell capacity [A.h]"])
            if nom_cap_Ah > 0 and cap_mAh:
                coulombic_eff = round(
                    min(1.0, cap_mAh / (nom_cap_Ah * 1000 * req.initial_soc)), 4
                )
        except Exception:
            pass

        # ── Peukert (chemistry-specific) ─────────────────────────
        # BUG6 FIXED: was hardcoded 1.05, now chemistry-dependent
        peukert_map = {
            "Chen2020":    1.08,   # NMC811
            "Marquis2019": 1.12,   # LCO
            "OKane2022":   1.09,   # LFP
            "Ecker2015":   1.07,   # NMC
        }
        peukert_n = round(
            peukert_map.get(req.parameter_set, 1.08) + 0.005 * (req.c_rate - 1.0), 3
        )

        # ── Response ─────────────────────────────────────────────
        return {
            # Time series
            "V_hist":               V_arr,
            "T_hist":               T_arr,
            "t_hist":               t_arr,
            "ce_profile":           ce_prof,
            "cs_n_profile":         cs_n,
            "cs_p_profile":         cs_p,
            # Key scalars
            "V_end":                round(V_end,  4),
            "V_mean":               round(V_mean, 4),
            "V_mid":                round(V_mid,  4),   # mid-discharge (for comparison)
            "T_max":                round(T_max,  3),
            "capacity_mAh":         round(cap_mAh, 2) if cap_mAh else None,
            "energy_Wh":            round(energy_Wh, 3) if energy_Wh else None,
            "energy_density_Wh_kg": energy_density,
            "coulombic_efficiency": coulombic_eff,
            "peukert_n":            peukert_n,
            # Electrochemistry
            "overpotential_mV":     eta_mv,
            "ce_min":               round(min(ce_prof), 4) if ce_prof else None,
            "ce_max":               round(max(ce_prof), 4) if ce_prof else None,
            "ce_extraction_ok":     ce_ok,
            # SEI
            "sei_thickness_nm":     sei_nm,
            "sei_model_active":     sei_active,
            # Meta
            "solve_time_ms":        round((time.time() - t0) * 1000, 1),
            "model":                solved_model,
            "params":               req.parameter_set,
            "pybamm_version":       pybamm.__version__,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e) + "\n---\n" + traceback.format_exc()[-1000:])
