#!/usr/bin/env python3
"""
Test rapid fix quantum — verifică că cost_fn este diferențiabilă.
Rulează: python3 ~/Desktop/Aladin/test_quantum_fix.py
"""
import numpy as np

try:
    import pennylane as qml
    from pennylane import numpy as qnp
except ImportError:
    print("❌ PennyLane nu e instalat: pip install pennylane")
    exit(1)

print("⚛️  Test Quantum Fix...")
print(f"   PennyLane version: {qml.__version__}")

dev = qml.device("default.qubit", wires=6)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs * np.pi, wires=range(6))
    qml.StronglyEntanglingLayers(weights, wires=range(6))
    return qml.expval(qml.PauliZ(0))

rng = np.random.default_rng(42)
weights = qnp.array(rng.uniform(-np.pi/4, np.pi/4, size=(2, 6, 3)), requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.03)

# Simulăm un batch de 5 samples
pre_inp = [np.array([0.5, 0.3, 1.0, 0.0, 0.5, 1.0], dtype=float) for _ in range(5)]
pre_tgt = [7.0/64.0, 2.0/64.0, 7.0/64.0, 2.0/64.0, 7.0/64.0]

def cost_fn(w):
    _avg_inp = qnp.array(np.mean(pre_inp, axis=0), requires_grad=False)
    _avg_tgt = float(np.mean([1.0 if t > 0.1 else -1.0 for t in pre_tgt]))
    _pred = circuit(_avg_inp, w)
    return (_pred - _avg_tgt) ** 2

errors = []
for epoch in range(5):
    try:
        weights, loss_val = opt.step_and_cost(cost_fn, weights)
        print(f"   ✅ Epoch {epoch}/5 — loss: {float(loss_val):.6f}")
    except Exception as e:
        print(f"   ❌ Epoch {epoch} EROARE: {e}")
        errors.append(str(e))

if not errors:
    print("\n✅ QUANTUM FIX OK — poți rula train_mario_ai.py")
else:
    print(f"\n❌ Erori: {errors}")
