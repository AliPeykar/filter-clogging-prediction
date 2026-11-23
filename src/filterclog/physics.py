"""Physical formulation of gas flow through the cartridge filter.

Measured quantities and units (fundamental to the problem):

* ``dp``       — pressure drop across the filter, Delta-P [kPa]
* ``flowrate`` — volumetric flow rate, Q [m^3/h]

Governing relation (thesis, Ch. 3): compressible flow through a porous
cartridge follows the Darcy–Forchheimer law. For fixed geometry it reduces to

    Delta-P = A * Q + B * Q^2                                   (Forchheimer)

where the *viscous* coefficient A ∝ mu / k1 (k1 = Darcy permeability) and the
*inertial* coefficient B ∝ rho / k2 (k2 = non-Darcy permeability). Clogging
lowers k1 and k2, so both A and B grow over the filter's life.

Identifiability on this dataset (verified empirically, see
docs/physics_formulation.md): the lab protocol holds Q essentially constant
(median within-window std of Q ~= 0.03 m^3/h), so Q and Q^2 are collinear
inside any causal window — a rolling Forchheimer fit is ill-conditioned
(condition numbers 1e2–1e3) and yields unphysical B < 0. Consequently:

* The per-row identifiable physical state variable is the **specific flow
  resistance**  R(t) = Delta-P / Q. In the Darcy regime R ∝ mu·L_eff/k1(t),
  i.e. R(t) tracks the reciprocal permeability directly; its growth rate
  dR/dt is the instantaneous fouling rate. R and its causal dynamics are
  model features.
* The Forchheimer coefficients are estimated **globally per lifecycle
  phase** (the early healthy segment contains flow excursions up to
  22 m^3/h) and reported as descriptive physics — an EDA/paper figure, not a
  model feature.

Unit discipline: all SI conversions are exact constants below. Note that
tree ensembles are invariant to strictly monotone rescalings, so converting a
feature from mixed units (kPa·h/m^3) to SI (Pa·s/m^3) cannot change any
split, prediction, or importance — the conversion matters for physical
interpretation, figures, and reported tables, and is therefore applied there.
"""

from __future__ import annotations

import numpy as np

KPA_TO_PA = 1.0e3
M3H_TO_M3S = 1.0 / 3600.0
#: (dp[kPa] / Q[m^3/h]) * RESISTANCE_SI = R in Pa·s/m^3
RESISTANCE_SI = KPA_TO_PA / M3H_TO_M3S  # = 3.6e6


def si_resistance(dp_kpa: np.ndarray, q_m3h: np.ndarray, q_floor: float = 0.5) -> np.ndarray:
    """Specific flow resistance in SI units [Pa·s/m^3].

    R = Delta-P / Q; in the Darcy regime R ∝ 1/k1 (reciprocal permeability).
    ``q_floor`` guards against division blow-up at sensor dropouts.
    """
    return (np.asarray(dp_kpa, float) / np.clip(q_m3h, q_floor, None)) * RESISTANCE_SI


def fit_forchheimer(
    q_m3h: np.ndarray,
    dp_kpa: np.ndarray,
    with_intercept: bool = False,
) -> dict:
    """Least-squares Forchheimer fit  Delta-P = A·Q + B·Q^2 (+ C).

    Intended for *global / per-phase* estimation where Q genuinely varies;
    the result carries its own diagnostics so ill-conditioned fits are
    self-documenting. Coefficients are reported both in measurement units
    (kPa per (m^3/h)^n) and SI (Pa per (m^3/s)^n).
    """
    q = np.asarray(q_m3h, float)
    dp = np.asarray(dp_kpa, float)
    cols = [q, q**2] + ([np.ones_like(q)] if with_intercept else [])
    M = np.column_stack(cols)
    coef, residuals, *_ = np.linalg.lstsq(M, dp, rcond=None)
    pred = M @ coef
    ss_res = float(((dp - pred) ** 2).sum())
    ss_tot = float(((dp - dp.mean()) ** 2).sum()) or 1e-12
    a_kpa, b_kpa = float(coef[0]), float(coef[1])
    out = {
        "A_kpa_per_m3h": a_kpa,
        "B_kpa_per_m3h2": b_kpa,
        "A_si_pa_per_m3s": a_kpa * KPA_TO_PA / M3H_TO_M3S,
        "B_si_pa_per_m3s2": b_kpa * KPA_TO_PA / M3H_TO_M3S**2,
        "intercept_kpa": float(coef[2]) if with_intercept else 0.0,
        "r2": 1.0 - ss_res / ss_tot,
        "condition_number": float(np.linalg.cond(M)),
        "q_std": float(q.std()),
        "n": int(len(q)),
        "physical": bool(a_kpa >= 0 and b_kpa >= 0),
    }
    return out


def forchheimer_curve(fit: dict, q_grid: np.ndarray) -> np.ndarray:
    """Delta-P [kPa] predicted by a fit over a Q grid [m^3/h] (for figures)."""
    return (
        fit["A_kpa_per_m3h"] * q_grid
        + fit["B_kpa_per_m3h2"] * q_grid**2
        + fit["intercept_kpa"]
    )
