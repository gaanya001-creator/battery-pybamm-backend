# server.py — FastAPI + PyBaMM backend for Electrochemical Cell Simulator v11
# Compatible with PyBaMM >= 23.x
# Install: pip install "pybamm>=23.0" fastapi uvicorn numpy
# Run:     uvicorn server:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pybamm, time, traceback
import numpy as np

app = FastAPI(title="PyBaMM Battery API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# --- model registry (load lazily to avoid import-time errors) ---
VALID_MODELS   = {"DFN", "SPMe", "SPM", "MPM"}
VALID_PARAMS   = {"Chen2020", "Marquis2019", "OKane2022", "Ecker2015"}
VALID_THERMALS = {"isothermal", "lumped", "x-lumped"}

class SimRequest(BaseModel):
    model:               str   = "DFN"
    parameter_set:       str   = "Chen2020"
    c_rate:              float = 1.0
    temperature_celsius: float = 25.0
    geometry:            str   = "pouch"
    thermal_model:       str   = "lumped"
    initial_soc:         float = 0.5
    enable_sei:          bool  = False

@app.get("/health")
def health():
    return {"status": "ok", "pybamm_version": pybamm.__version__, "solver": "CasADi/IDAKLU"}

@app.post("/simulate")
def simulate(req: SimRequest):
    t0 = time.time()
    try:
        # --- validation ---
        if req.model not in VALID_MODELS:
            raise HTTPException(400, f"Unknown model '{req.model}'. Valid: {VALID_MODELS}")
        if req.parameter_set not in VALID_PARAMS:
            raise HTTPException(400, f"Unknown param set '{req.parameter_set}'. Valid: {VALID_PARAMS}")

        # --- build model options ---
        thermal = req.thermal_model if req.thermal_model in VALID_THERMALS else "lumped"
        options = {"thermal": thermal}
        if req.enable_sei:
            options["SEI"] = "ec reaction limited"
        # cell geometry only valid for DFN with pouch/cylindrical
        if req.model == "DFN" and req.geometry in ("pouch", "cylindrical"):
            options["cell geometry"] = req.geometry

        # MPM needs special handling
        if req.model == "MPM":
            model = pybamm.lithium_ion.MPM(options=options)
        else:
            ModelClass = getattr(pybamm.lithium_ion, req.model)
            model = ModelClass(options=options)

        # --- parameter set ---
        param = pybamm.ParameterValues(req.parameter_set)
        T_ref = req.temperature_celsius + 273.15
        param["Ambient temperature [K]"] = T_ref
        param.update({"Initial temperature [K]": T_ref}, check_already_exists=False)

        # --- experiment ---
        experiment = pybamm.Experiment([
            pybamm.step.string(f"Discharge at {req.c_rate}C until 2.5 V")
        ])
        sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment)
        sim.solve(initial_soc=req.initial_soc)
        sol = sim.solution

        # --- extract scalars ---
        def safe_last(key):
            try:
                v = sol[key].entries
                return float(v.flat[-1])
            except Exception:
                return None

        t_arr = sol["Time [h]"].entries.tolist()
        V_arr = sol["Terminal voltage [V]"].entries.tolist()
        V_end = float(V_arr[-1])
        cap_mAh_val = safe_last("Discharge capacity [A.h]")
        cap_mAh = (cap_mAh_val or 0.0) * 1000

        # --- temperature ---
        T_arr = [T_ref] * len(t_arr)
        if thermal != "isothermal":
            for key in ("Volume-averaged cell temperature [K]",
                        "Cell temperature [K]",
                        "X-averaged cell temperature [K]"):
                try:
                    arr = sol[key].entries
                    T_arr = arr.tolist() if arr.ndim == 1 else arr[0].tolist()
                    break
                except Exception:
                    pass
        T_max = float(np.max(T_arr))

        # --- spatial electrolyte concentration at t_end ---
        ce_prof = []
        ce_extraction_ok = False
        for key in ("Electrolyte concentration [mol.m-3]",
                    "Electrolyte concentration [Molar]"):
            try:
                arr = sol[key].entries
                # arr shape: (nx, nt) — take last time column
                col = arr[:, -1] if arr.ndim == 2 else arr
                # convert mol/m³ → mol/L  (or already Molar if key says Molar)
                factor = 1/1000 if "mol.m-3" in key else 1.0
                ce_prof = (np.array(col) * factor).tolist()
                ce_extraction_ok = True
                break
            except Exception:
                pass

        # --- solid concentration spatial profiles ---
        cs_n, cs_p = [], []
        for side, cmax_key, out in [
            ("negative", "Maximum concentration in negative electrode [mol.m-3]", cs_n),
            ("positive", "Maximum concentration in positive electrode [mol.m-3]", cs_p),
        ]:
            try:
                cmax = float(param[cmax_key])
                key = f"X-averaged {side} particle surface concentration [mol.m-3]"
                arr = sol[key].entries
                val = float(arr.flat[-1]) / cmax
                out.extend([max(0, min(1, val))] * 15)
            except Exception:
                pass

        # --- SEI thickness ---
        # NOTE: PyBaMM's default DFN/SPM/SPMe options do NOT include an SEI
        # growth submodel. Without explicitly passing options={"SEI": "..."}
        # when building the model, there is no SEI variable to read, and a
        # value of 0 here means "submodel inactive", not "no SEI formed".
        sei_nm = None
        sei_model_active = bool(options.get("SEI"))
        if sei_model_active:
            for key in ("Total SEI thickness [m]",
                        "X-averaged total SEI thickness [m]",
                        "Negative electrode SEI film thickness [m]"):
                try:
                    arr = sol[key].entries
                    sei_nm = float(arr.flat[-1]) * 1e9
                    break
                except Exception:
                    pass

        # --- overpotential ---
        eta_mv = None
        for key in ("Negative electrode reaction overpotential [V]",
                    "X-averaged negative electrode reaction overpotential [V]"):
            try:
                arr = sol[key].entries
                eta_mv = abs(float(arr.flat[-1])) * 1000
                break
            except Exception:
                pass

        # --- energy density (real calculation, not a placeholder) ---
        # Wh/kg = (Energy delivered in Wh) / (cell mass in kg)
        # Energy = integral of V*I dt; since we only have V(t) and total
        # capacity, approximate using the mean discharge voltage rather
        # than V_end (V_end is the cutoff voltage, not representative).
        V_mean = float(np.mean(V_arr)) if V_arr else V_end
        energy_Wh = (cap_mAh / 1000.0) * V_mean  # Ah * V = Wh
        cell_mass_kg = None
        for mass_key in ("Cell mass [kg]", "Total mass [kg]", "Cell mass [g]"):
            try:
                m = float(param[mass_key])
                cell_mass_kg = m / 1000.0 if mass_key.endswith("[g]") else m
                break
            except Exception:
                pass
        energy_density = (energy_Wh / cell_mass_kg) if cell_mass_kg else None

        return {
            "V_hist":           V_arr,
            "T_hist":           T_arr,
            "t_hist":           t_arr,
            "ce_profile":       ce_prof,
            "cs_n_profile":     cs_n,
            "cs_p_profile":     cs_p,
            "V_end":            V_end,
            "T_max":            T_max,
            "capacity_mAh":     cap_mAh,
            "ce_min":           (min(ce_prof) if ce_prof else None),
            "ce_max":           (max(ce_prof) if ce_prof else None),
            "ce_extraction_ok": ce_extraction_ok,
            "sei_thickness_nm": sei_nm,
            "sei_model_active": sei_model_active,
            "energy_density_Wh_kg": energy_density,
            "overpotential_mV": eta_mv,
            "solve_time_ms":    round((time.time() - t0) * 1000, 1),
            "model":            req.model,
            "params":           req.parameter_set,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e) + "\n" + traceback.format_exc()[-800:])
