import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pennylane as qml
from pennylane import numpy as pnp
from src.main import heft_va_ansatz, heft_va_init_fn

def run_lih_benchmark():
    print("--- H-EFT-VA v2: LiH 10-Qubit Scaling Test ---")
    symbols = ["Li", "H"]
    coords = pnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.595])
    
    # Active space: 2 electrons in 5 orbitals = 10 qubits
    H_mol, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols, coords, active_electrons=2, active_orbitals=5
    )
    hf_state = qml.qchem.hf_state(2, n_qubits)
    exact_energy = -7.676912 #
    
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def cost_fn(params):
        qml.BasisState(hf_state, wires=range(n_qubits))
        heft_va_ansatz(params, n_qubits, n_layers=1, mode='chemistry')
        return qml.expval(H_mol)

    init_params = heft_va_init_fn(n_qubits, n_layers=1, mode='chemistry')
    opt = qml.AdamOptimizer(stepsize=0.03)
    
    params = init_params
    for i in range(121):
        params, energy = opt.step_and_cost(cost_fn, params)
        if i % 30 == 0:
            print(f"Step {i}: Energy = {energy:.6f} Ha")

    print(f"\nFinal LiH Energy: {energy:.8f} Ha")
    print(f"Error: {abs(energy - exact_energy):.2e} Ha")

if __name__ == "__main__":
    run_lih_benchmark()
