"""Numerical checks of Secs. IV.B and V.B of the paper (inputs are the simulation results quoted there).

1. Poisson criterion of Sec. V.B: a per-round check pays off iff kappa_c < eta/(2-eta).
   Model: faults ~ Poisson(lambda), lambda = -ln ESP; the check multiplies lambda by (1+kappa) and detects each fault
   with probability eta, without correlation between faults; only fault-free runs carry signal; the cost of a shot does not depend on the
   circuit length (rep_delay dominated); Delta small so the per-shot variance is ~constant.
   => P(acc) = exp(-eta*lambda*(1+kappa)); QPU gain = exp(lambda*[2 - (2-eta)(1+kappa)]).
   Compared with the trajectory simulations of the N=15 uncomputation check (FakeBoston noise model with all error
   rates multiplied by s = 1, 2, 3; ESP of the 724-CZ circuit under the same model).
2. Effective noise scale s_eff of the N=15 run by linear interpolation of the FakeBoston x s simulations.
3. Ratios of signal retention: measured vs pre-run prediction and vs snapshot simulations.
"""
import math

# ---------------------------------------------------------------- 1. Poisson criterion
kappa = 890 / 724 - 1                      # relative CZ overhead of the check (724 -> 890 CZ)
cases = {  # s: (ESP of the 724-CZ circuit, P(acc) simulated, gain simulated (post-selection, 3 seeds), 68% interval)
    1: (0.206, 0.387, 1.41, (1.15, 1.72)),
    2: (0.043, 0.145, 1.24, (0.95, 1.57)),
    3: (0.009, 0.078, 1.02, (0.51, 1.77)),
}
print(f"kappa_c = {kappa:.3f}")
for s, (esp, pacc, g_sim, ci) in cases.items():
    lam = -math.log(esp)
    eta = -math.log(pacc) / (lam * (1 + kappa))
    thr = eta / (2 - eta)
    g_mod = math.exp(lam * (2 - (2 - eta) * (1 + kappa)))
    print(f"s={s}: lambda={lam:.2f}  eta (from P(acc))={eta:.3f}  threshold eta/(2-eta)={thr:.3f}  "
          f"criterion {'met' if kappa < thr else 'violated'}  gain model={g_mod:.2f}  simulated={g_sim:.2f} {ci}")

# ---------------------------------------------------------------- 2. s_eff
P_meas, se_meas = 0.28945, 0.00185
(s2, p2, e2), (s3, p3, e3) = (2, 0.299, 0.006), (3, 0.267, 0.005)
seff = s2 + (p2 - P_meas) / (p2 - p3) * (s3 - s2)
# crude range: move every input by +-1 sigma in the direction that widens the interval
lo = s2 + (p2 - e2 - (P_meas + se_meas)) / ((p2 - e2) - (p3 + e3))
hi = s2 + (p2 + e2 - (P_meas - se_meas)) / ((p2 + e2) - (p3 - e3))
print(f"s_eff = {seff:.2f}  (range {lo:.2f}-{hi:.2f});  required division for N=21 (threshold s~0.6): "
      f"{seff/0.6:.1f} (range {lo/0.6:.1f}-{hi/0.6:.1f})")

# ---------------------------------------------------------------- 3. retention ratios
P_id, P_rand = 0.5, 0.25
R = lambda p: (p - P_rand) / (P_id - P_rand)
Rm = R(P_meas)
for lab, p in (("pre-run prediction", 0.318), ("FakeKingston snapshot", 0.325), ("FakeBoston snapshot", 0.347)):
    print(f"{lab}: P={p}  R={R(p):.3f}  ratio to measured R={Rm:.3f}: {R(p)/Rm:.2f}")
