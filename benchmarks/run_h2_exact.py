import sys
import os
# Ensure the script can see the src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from src.main import heft_va_ansatz, heft_va_init_fn

def run_h2_benchmark():
    print("--- H-EFT-VA v2: H2 Numerical Exactness Test ---")
    symbols = ["H", "H"]
    coordinates = pnp.array([0.0, 0.0, -0.3714, 0.0, 0.0, 0.3714])
    H_mol, n_qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    hf_state = qml.qchem.hf_state(2, n_qubits)
    
    exact_energy = -0.899651 # Target Ground State
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def cost_fn(params):
        qml.BasisState(hf_state, wires=range(n_qubits))
        # Call with mode='chemistry' to use IsingXY entanglers
        heft_va_ansatz(params, n_qubits, n_layers=4, mode='chemistry')
        return qml.expval(H_mol)

    # Initialize using your EFT scaling rule: sigma = kappa / (L*N)
    init_params = heft_va_init_fn(n_qubits, n_layers=4, mode='chemistry')
    opt = qml.QNGOptimizer(stepsize=0.01) # Quantum Natural Gradient
    
    params = init_params
    for i in range(151):
        params, energy = opt.step_and_cost(cost_fn, params)
        if i % 50 == 0:
            print(f"Step {i}: Energy = {energy:.6f} Ha")

    print(f"\nFinal Energy: {energy:.8f} Ha")
    print(f"Error: {abs(energy - exact_energy):.2e} Ha")

if __name__ == "__main__":
    run_h2_benchmark()
