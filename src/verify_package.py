"""Self-test of the package (no IBM account needed).

1. Exactness: the noiseless line circuit (static version, final inverse QFT added) is simulated with Aer
   (statevector, one shot per seed = one outcome of the mid-circuit window measurement) and its exponent-register
   distribution is compared with the exact arithmetic model (total variation distance, up to the sign
   convention y -> -y); the executable semiclassical circuit is sampled and compared with the model.
2. Gate counts: the circuits of the hardware runs (true circuit and controls, N = 15 and N = 21) are built,
   placed on a line of an IBM noise model (FakeKingston and FakeFez) and transpiled; the CZ counts are printed
   (expected: 724 for N = 15, a = 7, t = 3 and 1538 for N = 21, a = 2, t = 5).
3. Ideal success of every circuit under criterion C2 (expected: N = 15: 0.5000 / 0.2500 / 0.2500 / 0.1250;
   N = 21: 0.2008 for the true circuit).

Usage: python src/verify_package.py [--quick]
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator

import exact_model as EM
import generator_core as GC
import run_ibm as RI
import shor_line_generator as LG
from hardware import HW

SIM = AerSimulator(method="statevector")


def tv_refl(P, Q):
    return min(0.5 * np.abs(P - Q).sum(), 0.5 * np.abs(P - np.roll(Q[::-1], 1)).sum())


def static_check(N, a, t, g, opt, seeds=(1, 2, 3, 4)):
    qc, ln = LG.build(N, a, t, g, opt, measure=False, return_line=True)
    ie = [ln.pos[f"e{k}"] for k in range(t)]
    qc.append(QFTGate(t).inverse(), ie)
    qc.save_probabilities(ie, label="p")
    tq = transpile(qc, SIM, optimization_level=0)
    Pm = EM.shor_P(N, a, t, g, order=opt.order, consts=opt.consts, table=opt.table, decohere=opt.decohere)
    return max(tv_refl(np.asarray(SIM.run(tq, shots=1, seed_simulator=s).result().data(0)["p"]), Pm)
               for s in seeds)


def semi_check(N, a, t, g, opt, shots):
    qc = LG.build(N, a, t, g, opt, measure=True)
    tq = transpile(qc, SIM, optimization_level=0)
    cnt = SIM.run(tq, shots=shots, seed_simulator=11).result().get_counts()
    Py = np.zeros(1 << t)
    for k, v in cnt.items():
        Py[int(k.split()[-2] if opt.decohere else k.split()[0], 2)] += v
    Py /= shots
    Pm = EM.shor_P(N, a, t, g, order=opt.order, consts=opt.consts, table=opt.table, decohere=opt.decohere)
    return tv_refl(Py, Pm)


def main():
    quick = "--quick" in sys.argv
    ok = True
    g = replace(GC.BASE_GENOME, m0=1, fill=0, G0=0)
    print("1. Exactness against the arithmetic model")
    cases = [(15, 7, 3), (15, 2, 4)]
    for N, a, t in cases:
        for name, opt in (("true", LG.Opt()), ("decohered e_{t-1}", LG.Opt(decohere=(t - 1,))),
                          ("scrambled constants",
                           LG.Opt(consts=EM.fake_consts(N, a, t, min(GC.lookup_bits(N, a, t), 2))))):
            t0 = time.time()
            tv = static_check(N, a, t, g, opt)
            good = tv < 1e-6          # floating-point accuracy of the fused two-qubit unitaries
            ok &= good
            print(f"   N={N} a={a} t={t} {name:22s} TV(Aer statevector, model) = {tv:.1e} "
                  f"{'OK' if good else 'FAIL'} [{time.time() - t0:.0f}s]")
    shots = 4000 if quick else 20000
    tv = semi_check(15, 7, 3, g, LG.Opt(semi=1), shots)
    good = tv < 4 * np.sqrt(8 / shots)
    ok &= good
    print(f"   N=15 a=7 t=3 semiclassical (mid-circuit measurements, feed-forward): TV({shots} shots, model) = "
          f"{tv:.3f} (statistical noise ~{np.sqrt(8 / shots) / 2:.3f}) {'OK' if good else 'FAIL'}")

    print("2. CZ counts after placement on a hardware line (no routing)")
    expected = {15: 724, 21: 1538}
    for be in ("FakeKingston", "FakeFez"):
        hw = HW(be)
        for N in (15, 21):
            r = RI.SETTINGS[N]
            t0 = time.time()
            t, g_, path, info = RI.prepare(N, hw, r["a"], r["t"])
            cz = {k: v["cz"] for k, v in info.items()}
            good = cz["true"] == expected[N]
            ok &= good
            print(f"   {be:12s} N={N} a={r['a']} t={t}: CZ {cz} (true circuit expected {expected[N]}: "
                  f"{'OK' if good else 'FAIL'}), {info['true']['tq'].num_qubits} device qubits, "
                  f"line of {len(path)} [{time.time() - t0:.0f}s]")
            ideal = {k: round(v["p_ideal"], 4) for k, v in info.items()}
            print(f"      ideal success (C2): {ideal}")
            if N == 15:
                good = ideal == {"true": 0.5, "classical_exponent": 0.25, "decohered_e2": 0.25,
                                 "scrambled_constants": 0.125}
            else:
                good = abs(ideal["true"] - 0.2008) < 1e-4
            ok &= good
        if quick:
            break
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
