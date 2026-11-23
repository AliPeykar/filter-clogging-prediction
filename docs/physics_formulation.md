# Physical Formulation — Gas Flow Through the Cartridge Filter

## Measured quantities and units

| Symbol | Column | Unit | Instrument |
|---|---|---|---|
| ΔP | `dp` | **kPa** | MPX5100DP differential-pressure sensor (0–100 kPa) |
| Q | `flowrate` | **m³/h** | rotameter |

Sampling: 11,519 rows over ≈ 11.5 h ⇒ ≈ 3.6 s/row (derived; archived in every
run's `metrics.json`).

## Governing physics

Flow through a porous cartridge obeys the **Darcy–Forchheimer** law. With
fixed geometry (effective area A_eff, medium thickness L):

  ΔP = A·Q + B·Q²

- **A ∝ μ / k₁** — viscous (Darcy) term; k₁ = Darcy permeability [m²]
- **B ∝ ρ / k₂** — inertial (non-Darcy) term; k₂ = inertial permeability [m]

Progressive clogging (black-powder deposition) reduces k₁ and k₂, so **A and
B increase monotonically over the filter's life** — the physical basis for
the monotone constraints used in the model (risk non-decreasing in
ΔP/resistance, non-increasing in flow).

### Unit conversions (exact)

| Quantity | Measurement units | SI | Factor |
|---|---|---|---|
| ΔP | kPa | Pa | × 10³ |
| Q | m³/h | m³/s | ÷ 3600 |
| R = ΔP/Q | kPa·h/m³ | Pa·s/m³ | × 3.6 × 10⁶ |
| A | kPa/(m³/h) | Pa/(m³/s) | × 3.6 × 10⁶ |
| B | kPa/(m³/h)² | Pa/(m³/s)² | × 1.296 × 10¹⁰ |

Implementation: `src/filterclog/physics.py` (constants `KPA_TO_PA`,
`M3H_TO_M3S`, `RESISTANCE_SI`).

## Identifiability analysis (why the features are what they are)

The lab protocol (thesis phase 2) holds Q essentially constant while powder
is injected. Empirically (diagnostic archived 2026-07-03):

- median within-window std of Q ≈ **0.03 m³/h** (windows 60–300 rows);
- a rolling Forchheimer regression is **ill-conditioned** (condition numbers
  10²–10³) and returns **B < 0 — unphysical** — because Q and Q² are
  collinear when Q barely moves.

Therefore:

1. **Per-row state variable = specific flow resistance**
   R(t) = ΔP/Q  [Pa·s/m³ in SI]. In the Darcy regime R ∝ μ·L/(k₁·A_eff), so
   R(t) tracks **reciprocal permeability** directly; it is the physically
   correct univariate clogging state under constant-flow operation. It is a
   model feature (`resistance`, with `resistance_ratio` = R/R_baseline as the
   dimensionless degradation index the thesis calls "normalized resistance").
2. **Fouling rate** dR/dt (trailing OLS slope of R) is the instantaneous
   permeability-decay rate — a new, physically motivated dynamic feature
   (`resistance_slope_15/30`). It is deliberately **unconstrained** in the
   monotone map: fouling rate is high during active clogging but returns
   toward zero on the clogged plateau, so its relation to the clogged *state*
   is non-monotone.
3. **Even per-phase global Forchheimer fits are confounded** — verified on
   this data: within any phase the ΔP variation is *degradation-driven*
   (dp rises with deposited mass at near-constant Q), not Q-driven, so
   regressing ΔP on Q inside a phase mixes the k(t) drift into the flow
   coefficients. Observed: R² < 0 and B < 0 in all three phases (archived in
   every run's `metrics.json` under `forchheimer_diagnostics`). The thesis
   avoids this on the industrial data via short (3-day) windows with velocity
   binning; the lab protocol provides no equivalent stationary-and-varying-Q
   regime. Consequently the physics figure (`dp_q_physics`) shows per-phase
   **Darcy operating lines** ΔP = R̄·Q with R̄ = median specific resistance —
   fully identifiable, and their rising slopes across phases exhibit the
   permeability decay directly (per-phase R̄ in SI archived under
   `phase_resistance_si`). `fit_forchheimer` remains in `physics.py` with
   self-documenting diagnostics for use on data where Q genuinely varies
   (e.g. the industrial SCADA campaign).

### Note on unit rescaling and tree models

Gradient-boosted trees are invariant to strictly monotone feature
transformations: converting R from kPa·h/m³ to Pa·s/m³ multiplies by a
positive constant and cannot change any split, prediction, ranking, or SHAP
ordering. Unit conversion therefore matters for **interpretation, figures,
and reported tables** — where SI values are used — while the design matrix
keeps measurement units. This is stated explicitly so reviewers do not
mistake unit bookkeeping for a modeling change.

## Dimensionless / normalized quantities used

| Feature | Definition | Physical meaning |
|---|---|---|
| `resistance_ratio` | R(t) / median R over first 20% of TRAIN | degradation index vs clean-filter baseline (thesis "normalized resistance") |
| `flow_efficiency` | Q / Q_max,train | capacity retention |
| `ema_spread` | EMA_fast(ΔP) − EMA_slow(ΔP) | short-vs-long trend divergence (thesis EMA criterion, causal re-computation) |
| `dp_cusum_pos` | one-sided CUSUM of z(ΔP), k = 0.5 | cumulative departure from healthy ΔP level (thesis CUSUM criterion, fixed-parameter causal form) |

Quantities requiring unmeasured inputs (gas density ρ, inlet pressure P_in,
filter area) — e.g. ΔP/(ρv²), ΔP/P_in from the industrial analysis — are
**not** computable for the lab dataset and are documented as such rather than
approximated with guessed constants.
