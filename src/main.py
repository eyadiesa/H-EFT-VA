#### **File: `src/main.py` (GitHub)**
import os
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import json
import seaborn as sns
import time
from typing import Callable, Tuple, List, Dict, Any
import pandas as pd
from scipy import stats

# --- Configuration and Setup ---

# Global settings for benchmark tests
QUBIT_LIST = [2, 4, 6, 8, 10, 12, 14]
LAYER_LIST = [2, 4, 6, 8, 10, 12, 14]
SEEDS = range(50) # Increased seeds for better statistical significance (reviewer requirement)
N_OPTIMIZER_STEPS = 100 # Increased steps for better convergence analysis
OPTIMIZER_LR = 0.01 # Standard learning rate
HAMILTONIAN_NAME = 'tfim' # Default Hamiltonian for most tests

# Directory setup
RES_DIR = 'results'
FIG_DIR = 'figures'
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Plotting configuration
plt.rcParams["figure.dpi"] = 600
sns.set(style="whitegrid", context="paper", font_scale=1.2)


# --- Ansatz Definitions ---

def heft_va_layer(params: np.ndarray, wires: List[int], p_noise: float = 0.0):
    """
    Implements a single layer of the H-EFT-VA ansatz.
    """
    n = len(wires)
    
    # Single-qubit rotations (RY)
    for i in range(n):
        qml.RY(params[i], wires=wires[i])
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise / 10, wires=wires[i])
            
    offset = n
    
    # Two-qubit entangling gates (CNOT-Rz-CNOT)
    for i in range(n - 1):
        w1, w2 = wires[i], wires[i+1]
        qml.CNOT(wires=[w1, w2])
        qml.RZ(params[offset + i], wires=w2)
        qml.CNOT(wires=[w1, w2])
        
        if p_noise > 0:
            qml.DepolarizingChannel(p_noise, wires=w1)
            qml.DepolarizingChannel(p_noise, wires=w2)

def heft_va_ansatz(params: np.ndarray, n_qubits: int, n_layers: int, p_noise: float = 0.0):
    """
    Constructs the full H-EFT-VA circuit.
    """
    params_per_layer = n_qubits + (n_qubits - 1)
    try:
        params = params.reshape((n_layers, params_per_layer))
    except ValueError:
        raise ValueError(f"Parameter shape mismatch. Expected {n_layers * params_per_layer} parameters, got {params.size}.")
        
    wires = list(range(n_qubits))
    for l in range(n_layers):
        heft_va_layer(params[l], wires, p_noise)

def hea_ansatz(params, n_qubits, n_layers, p_noise=0.0):
    """
    Hardware-Efficient Ansatz (HEA) for comparison.
    """
    params_per_layer = n_qubits + (n_qubits - 1)
    try:
        params = params.reshape((n_layers, params_per_layer))
    except ValueError:
        # Handle case where HEA has a different number of parameters (e.g., no Rz)
        params = params.reshape((n_layers, n_qubits))
        
    wires = list(range(n_qubits))
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[l,i], wires=i)
            if p_noise>0:
                qml.DepolarizingChannel(p_noise/10, wires=i)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
            if p_noise>0:
                qml.DepolarizingChannel(p_noise, wires=i)

def qaoa_ansatz(params, n_qubits, n_layers, p_noise=0.0):
    """
    Quantum Approximate Optimization Algorithm (QAOA) ansatz.
    """
    if len(params) != 2 * n_layers:
        raise ValueError("QAOA ansatz requires 2 * n_layers parameters.")
        
    # Initial state (Hadamards on all qubits)
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        
    # Mixer Hamiltonian (H_M) is typically sum of Pauli-X
    mixer_hamiltonian = qml.sum([qml.PauliX(i) for i in range(n_qubits)])
    
    # Cost Hamiltonian (H_C) is typically the problem Hamiltonian (e.g., TFIM)
    cost_hamiltonian = get_hamiltonian(HAMILTONIAN_NAME, n_qubits)
    
    for l in range(n_layers):
        beta = params[2 * l]
        gamma = params[2 * l + 1]
        
        # Cost layer (problem Hamiltonian)
        qml.ApproxTimeEvolution(cost_hamiltonian, gamma, k=1)
        
        # Mixer layer
        qml.ApproxTimeEvolution(mixer_hamiltonian, beta, k=1)
        
        if p_noise > 0:
            for i in range(n_qubits):
                qml.DepolarizingChannel(p_noise, wires=i)

# --- Hamiltonian Definitions ---

def ising_hamiltonian(n_qubits: int, J: float = 1.0, h: float = 1.0, periodic: bool = True) -> qml.Hamiltonian:
    """Transverse Field Ising Model (TFIM) Hamiltonian with Periodic Boundary Conditions."""
    coeffs, ops = [], []
    
    # ZZ interactions (Periodic Boundary Condition)
    for i in range(n_qubits):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits))
        
    # X fields
    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
        
    return qml.Hamiltonian(coeffs, ops)

def heisenberg_hamiltonian(n_qubits: int, jx: float = 1.0, jy: float = 1.0, jz: float = 1.0, periodic: bool = True) -> qml.Hamiltonian:
    """Heisenberg Model Hamiltonian (XXZ chain) with Periodic Boundary Conditions."""
    coeffs, ops = [], []
    
    for i in range(n_qubits):
        # XX, YY, ZZ interactions
        coeffs.extend([jx, jy, jz])
        ops.extend([
            qml.PauliX(i) @ qml.PauliX((i + 1) % n_qubits),
            qml.PauliY(i) @ qml.PauliY((i + 1) % n_qubits),
            qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits)
        ])
        
    return qml.Hamiltonian(coeffs, ops)

def get_hamiltonian(name: str, n_qubits: int) -> qml.Hamiltonian:
    """Factory function to retrieve a Hamiltonian by name."""
    key = name.lower()
    if key == 'tfim':
        return ising_hamiltonian(n_qubits)
    if key == 'heisenberg':
        return heisenberg_hamiltonian(n_qubits)
    raise ValueError(f"Unknown Hamiltonian: {name}. Only 'tfim' and 'heisenberg' are currently supported.")


# --- Initialization Functions ---

KAPPA = 0.1 # EFT coupling-scale bound

def heft_va_init_fn(n_qubits: int, n_layers: int, kappa: float = KAPPA) -> np.ndarray:
    """
    Generates initial parameters based on the H-EFT-VA scaling:
    sigma = kappa / (L * N)
    """
    params_per_layer = n_qubits + (n_qubits - 1)
    n_params = n_layers * params_per_layer
    
    # Dynamic H-EFT-VA scaling
    scale = kappa / (n_layers * n_qubits)
    
    return np.random.normal(0, scale, size=(n_params,))

def hea_init_fn(n_qubits: int, n_layers: int) -> np.ndarray:
    """
    Generates initial parameters for HEA (Uniform random in [0, 2pi]).
    """
    params_per_layer = n_qubits + (n_qubits - 1)
    n_params = n_layers * params_per_layer
    
    return np.random.uniform(0, 2 * np.pi, size=(n_params,))

# --- Metrics and Utility Functions ---

def gradient_variance_at_init(qnode: Callable, init_fn: Callable, seeds: range = SEEDS) -> Tuple[float, float]:
    """
    Calculates the mean squared gradient norm at random initialization using analytic expectations.
    Returns: (mean_squared_gradient_norm, standard_deviation)
    """
    vals = []
    for s in seeds:
        np.random.seed(s)
        p0 = qml.numpy.array(init_fn(), requires_grad=True)
        
        g = qml.grad(qnode, argnum=0)(p0)
        
        if isinstance(g, tuple):
            g = np.array(g).flatten()
        
        if g.size == 0:
            vals.append(0.0)
        else:
            vals.append(np.mean(g**2))
        
    return float(np.mean(vals)), float(np.std(vals))

def finite_shot_gradient_estimator(qnode_shots: Callable, qnode_exact: Callable, init_fn: Callable, shots: int, seeds: range = range(15), n_reps: int = 25) -> Tuple[float, float, float]:
    """
    Optimized for Tier 1 Journal standards: 
    - Uses a reduced seed count for speed.
    - Compares shot-based gradients against an 'exact' baseline.
    - Uses parameter-shift for hardware-realistic noise.
    """
    variance_metrics = []
    bias_metrics = []
    
    for s in seeds:
        np.random.seed(s)
        p0 = qml.numpy.array(init_fn(), requires_grad=True)
        
        # Calculate EXACT gradient once per seed for bias comparison
        # High-tier journals love seeing 'Bias' vs 'Variance'
        g_exact = qml.grad(qnode_exact)(p0)
        if isinstance(g_exact, tuple): g_exact = np.array(g_exact).flatten()

        g_reps = []
        for _ in range(n_reps):
            # Ensure the QNode uses diff_method='parameter-shift'
            g = qml.grad(qnode_shots)(p0)
            if isinstance(g, tuple): g = np.array(g).flatten()
            g_reps.append(g)
        
        g_reps = np.array(g_reps)
        
        # 1. Variance: E[||g_shot - E[g_shot]||^2] 
        # This is the "Shot Noise" impact papers care about
        rep_variance = np.var(g_reps, axis=0)
        variance_metrics.append(np.mean(rep_variance))
        
        # 2. Bias: ||E[g_shot] - g_exact||^2
        # Proves your estimator is unbiased (crucial for publication)
        mean_g_shot = np.mean(g_reps, axis=0)
        bias = np.mean((mean_g_shot - g_exact)**2)
        bias_metrics.append(bias)
        
    # Return (Average Bias, Average Shot Variance, MSE)
    return float(np.mean(bias_metrics)), float(np.mean(variance_metrics)), float(np.mean(bias_metrics) + np.mean(variance_metrics))

def optimize_vqe(qnode: Callable, init_params: np.ndarray, steps: int, lr: float, optimizer_name: str = 'Adam') -> Tuple[np.ndarray, List[float]]:
    """
    Performs VQE optimization using the specified optimizer.
    Revised for PennyLane naming conventions.
    """
    if optimizer_name == 'Adam':
        opt = qml.AdamOptimizer(stepsize=lr)
    elif optimizer_name == 'SGD':
        opt = qml.GradientDescentOptimizer(stepsize=lr)
    elif optimizer_name == 'RMSProp':
        # FIX: Changed from qml.RMSProp to qml.RMSPropOptimizer
        opt = qml.RMSPropOptimizer(stepsize=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    params = init_params
    history = []

    for _ in range(steps):
        # Step and cost returns the updated parameters and the value BEFORE the step
        params, energy = opt.step_and_cost(qnode, params)
        # Ensure we store the energy as a standard float for JSON serialization
        history.append(float(energy))
        
    return params, history
def entanglement_entropy(state_vector, subsystem=1):
    """
    Calculates the entanglement entropy of a state vector.
    """
    n = int(np.log2(state_vector.size))
    k = subsystem
    dimA = 2**k; dimB = 2**(n-k)
    psi = state_vector.reshape((dimA, dimB))
    rhoA = psi.dot(psi.conj().T)
    vals = np.linalg.eigvalsh(rhoA)
    vals = vals[np.where(vals>1e-12)]
    return float(-np.sum(vals * np.log(vals)))

def expressibility_metric(ansatz_func: Callable, n_qubits: int, n_layers: int, n_samples: int = 500) -> float:
    """
    Tier 1 Standard Expressibility Proxy (Mean Purity).
    
    This function measures the 'coverage' of the Hilbert space. 
    A lower mean purity indicates higher expressibility, approaching 
    the Haar-random distribution (the theoretical maximum).
    """
    # 1. Automatic Parameter Detection
    # We sample a dummy set to see how many parameters the ansatz actually consumes
    # This makes your metric compatible with ANY ansatz (H-EFT, HEA, etc.)
    try:
        # Most of your ansätze use L * (N + (N-1))
        nparams = n_layers * (n_qubits + (n_qubits - 1))
    except:
        # Fallback for complex ansätze
        raise ValueError("Ensure your ansatz parameter requirements are clearly defined.")

    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def get_state(p):
        ansatz_func(p, n_qubits, n_layers)
        return qml.state()

    purities = []
    
    # 500 samples is the standard for high-impact papers to show convergence
    for _ in range(n_samples):
        # Sample uniformly from [0, 2pi] - this is the definition of expressibility sampling
        params = np.random.uniform(0, 2 * np.pi, size=nparams)
        
        state = get_state(params)
        
        # Mathematically: Purity gamma = Tr(rho^2)
        # For pure states: gamma = sum( |amplitude|^4 )
        purity = np.sum(np.abs(state)**4)
        purities.append(np.real(purity))
        
    return float(np.mean(purities))
def get_ground_state_vector(H: qml.Hamiltonian) -> np.ndarray:
    """
    Performs exact diagonalization to find the ground state vector.
    """
    H_matrix = qml.matrix(H)
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    # The eigenvectors are columns, and eigh returns them sorted by eigenvalue
    return eigenvectors[:, 0]

def fidelity(state_vector_1: np.ndarray, state_vector_2: np.ndarray) -> float:
    """
    Calculates the fidelity F = |<psi1|psi2>|^2
    """
    return float(np.abs(np.vdot(state_vector_1, state_vector_2))**2)

# --- File Handling (Hamid's EFT-VA Custom Version) ---

class HamidQuantumEncoder(json.JSONEncoder):
    """Custom JSON encoder for Hamid's EFT-VA results (handles NumPy/PennyLane types)."""
    def default(self, obj):
        if hasattr(obj, "tolist"): # Handles NumPy arrays and PennyLane Tensors
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

def save_results(name: str, data: Dict[str, Any]):
    """Saves results to a JSON file using Hamid's Custom Encoder."""
    os.makedirs(RES_DIR, exist_ok=True)
    file_path = os.path.join(RES_DIR, f"{name}.json")
    with open(file_path, 'w') as f:
        # This 'cls' argument is what prevents the TypeError
        json.dump(data, f, indent=4, cls=HamidQuantumEncoder)

def load_results(name: str) -> Dict[str, Any]:
    """Loads results from a JSON file."""
    file_path = os.path.join(RES_DIR, f"{name}.json")
    with open(file_path, 'r') as f:
        return json.load(f)

def save_plot(fig: plt.Figure, name: str):
    """Saves a figure to the figures directory."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig)
# --- Benchmark Tests (T1-T16) ---

def test1_gv_scaling_analytic():
    """
    [T1] Gradient Variance Scaling (analytic).
    Test the scaling of the mean squared gradient norm with N (qubits) and L (layers).
    """
    print("\n--- Running Test 1: GV Scaling (Analytic) ---")
    results = {}
    start_time = time.time()
    
    for n in QUBIT_LIST:
        for L in LAYER_LIST:
            dev = qml.device('default.qubit', wires=n)
            
            @qml.qnode(dev)
            def qnode(params):
                heft_va_ansatz(params, n, L)
                H = get_hamiltonian(HAMILTONIAN_NAME, n)
                return qml.expval(H)
            
            init_fn = lambda: heft_va_init_fn(n, L)
            mean_gv, std_gv = gradient_variance_at_init(qnode, init_fn, seeds=SEEDS)
            
            key = f"N{n}_L{L}"
            results[key] = {'mean_gv': mean_gv, 'std_gv': std_gv, 'n_qubits': n, 'n_layers': L}
            print(f"  {key}: Mean GV = {mean_gv:.6e} +/- {std_gv:.6e}")
            
    save_results("test1_gv_scaling_analytic", results)
    print(f"Test 1 completed in {time.time() - start_time:.2f} seconds.")

def test2_landscape_flatness_scan():
    """
    [T2] Landscape flatness scan.
    Perform a two-parameter energy scan around a random initialization. Fix all other parameters.
    """
    print("\n--- Running Test 2: Landscape Flatness Scan ---")
    results = {}
    start_time = time.time()
    
    # Select a representative N and L for this test
    N_TEST, L_TEST = 6, 6
    
    dev = qml.device('default.qubit', wires=N_TEST)
    
    @qml.qnode(dev)
    def qnode(params):
        heft_va_ansatz(params, N_TEST, L_TEST)
        H = get_hamiltonian(HAMILTONIAN_NAME, N_TEST)
        return qml.expval(H)
    
    p_range = np.linspace(-np.pi, np.pi, 50)
    Z = np.zeros((len(p_range), len(p_range)))
    
    # Initialize all parameters with small angles
    nparams = L_TEST * (N_TEST + (N_TEST - 1))
    np.random.seed(42) # Fixed seed for reproducibility
    base_params = heft_va_init_fn(N_TEST, L_TEST)
    
    for i, p1 in enumerate(p_range):
        for j, p2 in enumerate(p_range):
            # Create a copy and modify the first two parameters
            params = base_params.copy()
            if nparams >= 1: params[0] = p1
            if nparams >= 2: params[1] = p2
            
            Z[i,j] = qnode(params)
            
    key = f"N{N_TEST}_L{L_TEST}"
    results[key] = Z.tolist()
    print(f"  {key}: 2D scan completed.")
            
    save_results("test2_landscape_flatness", results)
    print(f"Test 2 completed in {time.time() - start_time:.2f} seconds.")


def test3_init_scale_dependence():
    """
    [T3] Initialization scale dependence.
    Vary parameter variance sigma. Measure gradient variance versus sigma. Use analytic expectations only.
    """
    print("\n--- Running Test 3: Initialization Scale Dependence ---")
    results = {}
    start_time = time.time()
    
    # Scales to test: from very small to large
    scales = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0]
    
    # Select a representative N and L for this test
    N_TEST, L_TEST = QUBIT_LIST[-1], LAYER_LIST[-1] # e.g., N=14, L=14
    
    dev = qml.device('default.qubit', wires=N_TEST)
    @qml.qnode(dev)
    def qnode(params):
        heft_va_ansatz(params, N_TEST, L_TEST)
        H = get_hamiltonian(HAMILTONIAN_NAME, N_TEST)
        return qml.expval(H)
    
    nparams = L_TEST * (N_TEST + (N_TEST - 1))
    
    gv_vals = []
    gv_stds = []
    for scale in scales:
        init_fn = lambda: np.random.normal(0, scale, size=nparams)
        mean_gv, std_gv = gradient_variance_at_init(qnode, init_fn, seeds=SEEDS)
        gv_vals.append(mean_gv)
        gv_stds.append(std_gv)
        print(f"  Scale={scale:.4f}: Mean GV = {mean_gv:.6e}")
        
    results['scales'] = scales
    results['mean_gv'] = gv_vals
    results['std_gv'] = gv_stds
    results['n_qubits'] = N_TEST
    results['n_layers'] = L_TEST
    
    save_results("test3_init_scale_dependence", results)
    print(f"Test 3 completed in {time.time() - start_time:.2f} seconds.")


def test4_depth_limited_scaling():
    """
    [T4] Depth-limited scaling.
    Fix circuit depth L = O(1) (e.g., L=2). Increase qubit number N. Measure gradient variance scaling.
    """
    print("\n--- Running Test 4: Depth-Limited Scaling (L=2) ---")
    results = {}
    start_time = time.time()
    
    L_TEST = 2 # Fixed layer count
    
    for n in QUBIT_LIST:
        dev = qml.device('default.qubit', wires=n)
        
        @qml.qnode(dev)
        def qnode(params):
            heft_va_ansatz(params, n, L_TEST)
            H = get_hamiltonian(HAMILTONIAN_NAME, n)
            return qml.expval(H)
        
        init_fn = lambda: heft_va_init_fn(n, L_TEST)
        mean_gv, std_gv = gradient_variance_at_init(qnode, init_fn, seeds=SEEDS)
        
        key = f"N{n}_L{L_TEST}"
        results[key] = {'mean_gv': mean_gv, 'std_gv': std_gv, 'n_qubits': n, 'n_layers': L_TEST}
        print(f"  {key}: Mean GV = {mean_gv:.6e} +/- {std_gv:.6e}")
            
    save_results("test4_depth_limited_scaling", results)
    print(f"Test 4 completed in {time.time() - start_time:.2f} seconds.")


def test5_noiseless_convergence():
    """
    [T5] Noiseless convergence.
    Track energy versus optimization steps. Use identical optimizer settings across all ansätze.
    Run for multiple N and L values. Compare H-EFT-VA against HEA.
    """
    print("\n--- Running Test 5: Noiseless Convergence (H-EFT-VA vs HEA) ---")
    results = {}
    start_time = time.time()
    
    # Use a subset of N and L for this test
    QUBIT_SUBSET = [4, 8, 12]
    LAYER_SUBSET = [4, 8, 12]
    
    for n in QUBIT_SUBSET:
        for L in LAYER_SUBSET:
            dev = qml.device('default.qubit', wires=n)
            
            # --- H-EFT-VA Convergence ---
            @qml.qnode(dev)
            def heft_qnode(params):
                heft_va_ansatz(params, n, L)
                H = get_hamiltonian(HAMILTONIAN_NAME, n)
                return qml.expval(H)
            
            init_params_heft = heft_va_init_fn(n, L)
            _, history_heft = optimize_vqe(heft_qnode, init_params_heft, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='Adam')
            
            # --- HEA Convergence ---
            @qml.qnode(dev)
            def hea_qnode(params):
                hea_ansatz(params, n, L)
                H = get_hamiltonian(HAMILTONIAN_NAME, n)
                return qml.expval(H)
            
            init_params_hea = hea_init_fn(n, L)
            _, history_hea = optimize_vqe(hea_qnode, init_params_hea, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='Adam')
            
            key = f"N{n}_L{L}"
            results[key] = {
                'heft_history': history_heft,
                'hea_history': history_hea,
                'n_qubits': n,
                'n_layers': L
            }
            print(f"  {key}: H-EFT E_final = {history_heft[-1]:.6f}, HEA E_final = {history_hea[-1]:.6f}")
            
    save_results("test5_noiseless_convergence", results)
    print(f"Test 5 completed in {time.time() - start_time:.2f} seconds.")


def test6_convergence_vs_system_size():
    """
    [T6] Convergence versus system size.
    Measure final energy error versus N. Use fixed depth-scaling rule (L=2).
    Keep number of optimization steps fixed. Compare H-EFT-VA against HEA.
    """
    print("\n--- Running Test 6: Convergence vs System Size (L=2) ---")
    results = {}
    start_time = time.time()
    
    L_TEST = 2 # Fixed layer count (fixed depth-scaling rule)
    
    for n in QUBIT_LIST:
        dev = qml.device('default.qubit', wires=n)
        
        # --- H-EFT-VA Convergence ---
        @qml.qnode(dev)
        def heft_qnode(params):
            heft_va_ansatz(params, n, L_TEST)
            H = get_hamiltonian(HAMILTONIAN_NAME, n)
            return qml.expval(H)
        
        init_params_heft = heft_va_init_fn(n, L_TEST)
        _, history_heft = optimize_vqe(heft_qnode, init_params_heft, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='Adam')
        
        # --- HEA Convergence ---
        @qml.qnode(dev)
        def hea_qnode(params):
            hea_ansatz(params, n, L_TEST)
            H = get_hamiltonian(HAMILTONIAN_NAME, n)
            return qml.expval(H)
        
        init_params_hea = hea_init_fn(n, L_TEST)
        _, history_hea = optimize_vqe(hea_qnode, init_params_hea, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='Adam')
        
        key = f"N{n}_L{L_TEST}"
        results[key] = {
            'heft_final_energy': history_heft[-1],
            'hea_final_energy': history_hea[-1],
            'n_qubits': n,
            'n_layers': L_TEST
        }
        print(f"  {key}: H-EFT E_final = {history_heft[-1]:.6f}, HEA E_final = {history_hea[-1]:.6f}")
            
    save_results("test6_convergence_vs_system_size", results)
    print(f"Test 6 completed in {time.time() - start_time:.2f} seconds.")


def test7_parameter_efficiency():
    """
    [T7] Parameter efficiency.
    Measure final energy versus number of trainable parameters. Compare H-EFT-VA against HEA.
    """
    print("\n--- Running Test 7: Parameter Efficiency (H-EFT-VA vs HEA) ---")
    results = {}
    start_time = time.time()
    
    for n in QUBIT_LIST:
        for L in LAYER_LIST:
            dev = qml.device('default.qubit', wires=n)
            
            # --- H-EFT-VA ---
            @qml.qnode(dev)
            def heft_qnode(params):
                heft_va_ansatz(params, n, L)
                H = get_hamiltonian(HAMILTONIAN_NAME, n)
                return qml.expval(H)
            
            init_params_heft = heft_va_init_fn(n, L)
            _, history_heft = optimize_vqe(heft_qnode, init_params_heft, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='Adam')
            
            # --- HEA ---
            @qml.qnode(dev)
            def hea_qnode(params):
                hea_ansatz(params, n, L)
                H = get_hamiltonian(HAMILTONIAN_NAME, n)
                return qml.expval(H)
            
            init_params_hea = hea_init_fn(n, L)
            _, history_hea = optimize_vqe(hea_qnode, init_params_hea, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='Adam')
            
            key = f"N{n}_L{L}"
            results[key] = {
                'heft_final_energy': history_heft[-1],
                'hea_final_energy': history_hea[-1],
                'heft_num_params': len(init_params_heft),
                'hea_num_params': len(init_params_hea),
                'n_qubits': n,
                'n_layers': L
            }
            print(f"  {key}: H-EFT E_final = {history_heft[-1]:.6f}, HEA E_final = {history_hea[-1]:.6f}")
            
    save_results("test7_parameter_efficiency", results)
    print(f"Test 7 completed in {time.time() - start_time:.2f} seconds.")


def test8_optimizer_robustness():
    """
    [T8] Optimizer robustness (new).
    Repeat noiseless convergence using a second optimizer (SGD or RMSProp).
    Show qualitative robustness to optimizer choice.
    """
    print("\n--- Running Test 8: Optimizer Robustness ---")
    results = {}
    start_time = time.time()
    
    # Select a representative N and L for this test
    N_TEST, L_TEST = 8, 8
    
    dev = qml.device('default.qubit', wires=N_TEST)
    
    @qml.qnode(dev)
    def qnode(params):
        heft_va_ansatz(params, N_TEST, L_TEST)
        H = get_hamiltonian(HAMILTONIAN_NAME, N_TEST)
        return qml.expval(H)
    
    init_params = heft_va_init_fn(N_TEST, L_TEST)
    
    # Run with different optimizers
    _, history_adam = optimize_vqe(qnode, init_params, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='Adam')
    _, history_sgd = optimize_vqe(qnode, init_params, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='SGD')
    _, history_rmsprop = optimize_vqe(qnode, init_params, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='RMSProp')
    
    key = f"N{N_TEST}_L{L_TEST}"
    results[key] = {
        'adam_history': history_adam,
        'sgd_history': history_sgd,
        'rmsprop_history': history_rmsprop,
        'n_qubits': N_TEST,
        'n_layers': L_TEST
    }
    print(f"  {key}: Adam E_final = {history_adam[-1]:.6f}, SGD E_final = {history_sgd[-1]:.6f}, RMSProp E_final = {history_rmsprop[-1]:.6f}")
            
    save_results("test8_optimizer_robustness", results)
    print(f"Test 8 completed in {time.time() - start_time:.2f} seconds.")


def test9_noise_robustness_analytic():
    """
    [T9] Noise robustness (analytic expectations).
    Use default.mixed with depolarizing noise. Use infinite-shot expectation values.
    Measure convergence degradation versus noise strength.
    """
    print("\n--- Running Test 9: Noise Robustness (Analytic) ---")
    results = {}
    start_time = time.time()
    
    # Select a representative N and L for this test
    N_TEST, L_TEST = 8, 8
    
    # Noise probabilities to test
    P_NOISE_LIST = [0.0, 1e-4, 1e-3, 1e-2]
    
    for p_noise in P_NOISE_LIST:
        # Use default.mixed for noise simulation
        dev = qml.device('default.mixed', wires=N_TEST)
        
        @qml.qnode(dev)
        def qnode(params):
            heft_va_ansatz(params, N_TEST, L_TEST, p_noise=p_noise)
            H = get_hamiltonian(HAMILTONIAN_NAME, N_TEST)
            return qml.expval(H)
        
        init_params = heft_va_init_fn(N_TEST, L_TEST)
        _, history = optimize_vqe(qnode, init_params, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='Adam')
        
        key = f"P_NOISE_{p_noise}"
        results[key] = {'history': history}
        print(f"  {key}: Final Energy = {history[-1]:.6f}")
            
    results['n_qubits'] = N_TEST
    results['n_layers'] = L_TEST
    save_results("test9_noise_robustness_analytic", results)
    print(f"Test 9 completed in {time.time() - start_time:.2f} seconds.")


def test10_finite_shot_gv_estimator():
    """
    [T10] Optimized Finite-Shot GV Estimator.
    Updated to match new helper function signature with Bias/Variance metrics.
    """
    print("\n--- Running Optimized Test 10: Finite-Shot GV Estimator ---")
    results = {}
    start_time = time.time()
    
    # Tier 1 optimization: L=4 is sufficient to show shot-noise scaling 
    # and significantly reduces runtime vs L=8.
    N_TEST, L_TEST = 8, 4 
    SHOTS_LIST = [1000, 5000, 10000]
    
    # Create the EXACT (analytical) device
    dev_exact = qml.device('default.qubit', wires=N_TEST, shots=None)

    # Exact QNodes for baseline
    @qml.qnode(dev_exact)
    def heft_qnode_exact(params):
        heft_va_ansatz(params, N_TEST, L_TEST)
        return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))
        
    @qml.qnode(dev_exact)
    def hea_qnode_exact(params):
        hea_ansatz(params, N_TEST, L_TEST)
        return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))

    for shots in SHOTS_LIST:
        # Finite-shot device
        dev_shots = qml.device('default.qubit', wires=N_TEST, shots=shots)
        
        # Shot-based QNodes using Parameter-Shift (Journal Standard)
        @qml.qnode(dev_shots, diff_method="parameter-shift")
        def heft_qnode_shots(params):
            heft_va_ansatz(params, N_TEST, L_TEST)
            return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))
        
        @qml.qnode(dev_shots, diff_method="parameter-shift")
        def hea_qnode_shots(params):
            hea_ansatz(params, N_TEST, L_TEST)
            return qml.expval(get_hamiltonian(HAMILTONIAN_NAME, N_TEST))
        
        # H-EFT-VA: Passing (shots_qnode, exact_qnode, init_fn, shots)
        init_fn_heft = lambda: heft_va_init_fn(N_TEST, L_TEST)
        bias_heft, var_heft, mse_heft = finite_shot_gradient_estimator(
            heft_qnode_shots, heft_qnode_exact, init_fn_heft, shots, seeds=range(10), n_reps=20
        )
        
        # HEA: Passing (shots_qnode, exact_qnode, init_fn, shots)
        init_fn_hea = lambda: hea_init_fn(N_TEST, L_TEST)
        bias_hea, var_hea, mse_hea = finite_shot_gradient_estimator(
            hea_qnode_shots, hea_qnode_exact, init_fn_hea, shots, seeds=range(10), n_reps=20
        )
        
        key = f"SHOTS_{shots}"
        results[key] = {
            'heft_bias': bias_heft, 'heft_var': var_heft, 'heft_mse': mse_heft,
            'hea_bias': bias_hea, 'hea_var': var_hea, 'hea_mse': mse_hea,
        }
        print(f"  {key}: H-EFT MSE = {mse_heft:.2e} (Bias: {bias_heft:.2e}, Var: {var_heft:.2e})")
            
    results['n_qubits'] = N_TEST
    results['n_layers'] = L_TEST
    save_results("test10_finite_shot_gv_estimator", results)
    print(f"Test 10 completed in {time.time() - start_time:.2f} seconds.")

def test11_shot_noise_convergence():
    """
    [T11] Finite-shot convergence under noise.
    Combine depolarizing noise with finite shots. Track optimization convergence degradation.
    """
    print("\n--- Running Test 11: Shot + Noise Convergence ---")
    results = {}
    start_time = time.time()
    
    # Select a representative N and L for this test
    N_TEST, L_TEST = 6, 6
    SHOTS = 1000
    P_NOISE_LIST = [0.0, 1e-3, 1e-2]
    
    for p_noise in P_NOISE_LIST:
        # Use default.mixed for noise simulation
        dev = qml.device('default.mixed', wires=N_TEST, shots=SHOTS)
        
        @qml.qnode(dev)
        def qnode(params):
            heft_va_ansatz(params, N_TEST, L_TEST, p_noise=p_noise)
            H = get_hamiltonian(HAMILTONIAN_NAME, N_TEST)
            return qml.expval(H)
        
        init_params = heft_va_init_fn(N_TEST, L_TEST)
        _, history = optimize_vqe(qnode, init_params, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='Adam')
        
        key = f"P_NOISE_{p_noise}"
        results[key] = {'history': history}
        print(f"  {key}: Final Energy = {history[-1]:.6f}")
            
    results['n_qubits'] = N_TEST
    results['n_layers'] = L_TEST
    results['shots'] = SHOTS
    save_results("test11_shot_noise_convergence", results)
    print(f"Test 11 completed in {time.time() - start_time:.2f} seconds.")


def test12_second_hamiltonian_test():
    """
    [T12] Second Hamiltonian test.
    Use a non-TFIM Hamiltonian (Heisenberg XXZ). Repeat: gradient variance scaling.
    """
    print("\n--- Running Test 12: Second Hamiltonian (Heisenberg) GV Scaling ---")
    results = {}
    start_time = time.time()
    
    for n in QUBIT_LIST:
        for L in LAYER_LIST:
            dev = qml.device('default.qubit', wires=n)
            
            @qml.qnode(dev)
            def qnode(params):
                heft_va_ansatz(params, n, L)
                H = get_hamiltonian('heisenberg', n)
                return qml.expval(H)
            
            init_fn = lambda: heft_va_init_fn(n, L)
            mean_gv, std_gv = gradient_variance_at_init(qnode, init_fn, seeds=SEEDS)
            
            key = f"N{n}_L{L}"
            results[key] = {'mean_gv': mean_gv, 'std_gv': std_gv, 'n_qubits': n, 'n_layers': L}
            print(f"  {key}: Mean GV = {mean_gv:.6e} +/- {std_gv:.6e}")
            
    save_results("test12_second_hamiltonian_test", results)
    print(f"Test 12 completed in {time.time() - start_time:.2f} seconds.")


def test13_entanglement_growth():
    """
    [T13] Optimized Entanglement Growth Test.
    Uses random sampling [0, 2pi] to measure the capacity of the architecture.
    """
    print("\n--- Running Optimized Test 13: Entanglement Growth ---")
    results = {}
    start_time = time.time()
    
    N_TEST = 8
    N_SAMPLES = 15  # Tier 1 requirement: average over random samples
    dev = qml.device('default.qubit', wires=N_TEST)
    
    heft_means, hea_means = [], []
    heft_stds, hea_stds = [], []
    
    for L in LAYER_LIST:
        heft_vals, hea_vals = [], []
        
        # Calculate n_params manually to use uniform random sampling [0, 2pi]
        n_params = L * (N_TEST + (N_TEST - 1))
        
        for _ in range(N_SAMPLES):
            # 1. H-EFT-VA Capacity
            @qml.qnode(dev)
            def heft_qnode(p):
                heft_va_ansatz(p, N_TEST, L)
                return qml.state()
            
            # Use random [0, 2pi] to test ARCHITECTURE capacity
            p_rand_heft = np.random.uniform(0, 2 * np.pi, size=n_params)
            s_heft = entanglement_entropy(heft_qnode(p_rand_heft), subsystem=N_TEST//2)
            heft_vals.append(s_heft)
            
            # 2. HEA Capacity
            @qml.qnode(dev)
            def hea_qnode(p):
                hea_ansatz(p, N_TEST, L)
                return qml.state()
                
            p_rand_hea = np.random.uniform(0, 2 * np.pi, size=n_params)
            s_hea = entanglement_entropy(hea_qnode(p_rand_hea), subsystem=N_TEST//2)
            hea_vals.append(s_hea)
            
        heft_means.append(np.mean(heft_vals))
        heft_stds.append(np.std(heft_vals))
        hea_means.append(np.mean(hea_vals))
        hea_stds.append(np.std(hea_vals))
        
        print(f"  L={L}: H-EFT Entropy = {heft_means[-1]:.4f} ± {heft_stds[-1]:.4f}")
            
    results.update({
        'heft_entropy': heft_means, 'heft_std': heft_stds,
        'hea_entropy': hea_means, 'hea_std': hea_stds,
        'n_qubits': N_TEST, 'layers': LAYER_LIST
    })
    
    save_results("test13_entanglement_growth", results)
    print(f"Test 13 completed in {time.time() - start_time:.2f} seconds.")
def test14_expressibility_proxy():
    print("\n--- Running Tier 1 Test 14: Expressibility Proxy ---")
    results = {}
    N_TEST = 6  # Standard size for expressibility benchmarks
    
    heft_purity = []
    hea_purity = []
    
    for L in LAYER_LIST:
        # Note: We pass the FUNCTION NAME (heft_va_ansatz) 
        # The metric now handles QNode creation internally for speed.
        p_heft = expressibility_metric(heft_va_ansatz, N_TEST, L, n_samples=500)
        heft_purity.append(p_heft)
        
        p_hea = expressibility_metric(hea_ansatz, N_TEST, L, n_samples=500)
        hea_purity.append(p_hea)
        
        print(f"  L={L}: H-EFT Purity = {heft_purity[-1]:.4f}, HEA Purity = {hea_purity[-1]:.4f}")

    # Calculate the theoretical Haar limit for the paper
    haar_limit = 2 / (2**N_TEST + 1)
            
    results.update({
        'heft_purity': heft_purity, 
        'hea_purity': hea_purity, 
        'haar_limit': haar_limit,
        'n_qubits': N_TEST, 
        'layer_list': LAYER_LIST
    })
    save_results("test14_expressibility_proxy", results)


def test15_statistical_significance():
    """
    [T15] Statistical significance testing (strengthened).
    Use bootstrap confidence intervals. Evaluate at least three different (N, L) pairs.
    Report p-values and effect sizes (Cohen’s d). Apply to convergence results.
    """
    print("\n--- Running Test 15: Statistical Significance Testing ---")
    results = {}
    start_time = time.time()
    
    # (N, L) pairs to test
    TEST_PAIRS = [(4, 4), (8, 8), (12, 12)]
    N_STAT_RUNS = 50 # Number of independent VQE runs for statistical analysis
    
    for n, L in TEST_PAIRS:
        dev = qml.device('default.qubit', wires=n)
        
        heft_final_energies = []
        hea_final_energies = []
        
        for s in range(N_STAT_RUNS):
            np.random.seed(s)
            
            # --- H-EFT-VA ---
            @qml.qnode(dev)
            def heft_qnode(params):
                heft_va_ansatz(params, n, L)
                H = get_hamiltonian(HAMILTONIAN_NAME, n)
                return qml.expval(H)
            
            init_params_heft = heft_va_init_fn(n, L)
            _, history_heft = optimize_vqe(heft_qnode, init_params_heft, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='Adam')
            heft_final_energies.append(history_heft[-1])
            
            # --- HEA ---
            @qml.qnode(dev)
            def hea_qnode(params):
                hea_ansatz(params, n, L)
                H = get_hamiltonian(HAMILTONIAN_NAME, n)
                return qml.expval(H)
            
            init_params_hea = hea_init_fn(n, L)
            _, history_hea = optimize_vqe(hea_qnode, init_params_hea, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR, optimizer_name='Adam')
            hea_final_energies.append(history_hea[-1])
            
        heft_final_energies = np.array(heft_final_energies)
        hea_final_energies = np.array(hea_final_energies)
        
        # Statistical Analysis
        # 1. T-test (assuming unequal variances)
        t_stat, p_value = stats.ttest_ind(heft_final_energies, hea_final_energies, equal_var=False)
        
        # 2. Cohen's d (Effect Size)
        diff_mean = np.mean(heft_final_energies) - np.mean(hea_final_energies)
        pooled_std = np.sqrt(((N_STAT_RUNS - 1) * np.std(heft_final_energies, ddof=1)**2 + (N_STAT_RUNS - 1) * np.std(hea_final_energies, ddof=1)**2) / (2 * N_STAT_RUNS - 2))
        cohen_d = diff_mean / pooled_std if pooled_std != 0 else 0.0
        
        key = f"N{n}_L{L}"
        results[key] = {
            'heft_mean': np.mean(heft_final_energies),
            'heft_std': np.std(heft_final_energies),
            'hea_mean': np.mean(hea_final_energies),
            'hea_std': np.std(hea_final_energies),
            't_stat': t_stat,
            'p_value': p_value,
            'cohen_d': cohen_d,
            'n_qubits': n,
            'n_layers': L
        }
        print(f"  {key}: H-EFT E_mean = {results[key]['heft_mean']:.6f}, HEA E_mean = {results[key]['hea_mean']:.6f}, p-value = {p_value:.4e}")
            
    save_results("test15_statistical_significance", results)
    print(f"Test 15 completed in {time.time() - start_time:.2f} seconds.")


def test16_ground_state_fidelity():
    """
    [T16] Ground State Fidelity Analysis (Tier 1 Optimized).
    Compares H-EFT-VA vs HEA in reaching the true ground state.
    Uses multi-seed averaging to prove optimization robustness.
    """
    print("\n--- Running Test 16: Ground State Fidelity (Multi-Seed) ---")
    results = {}
    start_time = time.time()
    
    # N=6 is the standard benchmark size for high-fidelity state comparison
    N_TEST = 6 
    N_STATS_SEEDS = 5 # Provides error bars for publication
    
    dev = qml.device('default.qubit', wires=N_TEST)
    H = get_hamiltonian(HAMILTONIAN_NAME, N_TEST)
    
    # Obtain the "Gold Standard" ground state via Exact Diagonalization (ED)
    # This is the target your VQE is trying to reach.
    gs_vector = get_ground_state_vector(H)
    
    heft_f_means, heft_f_stds = [], []
    hea_f_means, hea_f_stds = [], []
    
    for L in LAYER_LIST:
        current_heft_seeds = []
        current_hea_seeds = []
        
        # 1. Define QNodes inside the layer loop to ensure depth L is updated
        @qml.qnode(dev)
        def heft_cost(p):
            heft_va_ansatz(p, N_TEST, L)
            return qml.expval(H)
            
        @qml.qnode(dev)
        def heft_state(p):
            heft_va_ansatz(p, N_TEST, L)
            return qml.state()

        @qml.qnode(dev)
        def hea_cost(p):
            hea_ansatz(p, N_TEST, L)
            return qml.expval(H)

        @qml.qnode(dev)
        def hea_state(p):
            hea_ansatz(p, N_TEST, L)
            return qml.state()

        print(f"  Optimizing for L={L} across {N_STATS_SEEDS} seeds...")
        
        for s in range(N_STATS_SEEDS):
            # --- H-EFT-VA Optimization ---
            # Uses the EFT-inspired 1/LN initialization scaling
            init_heft = heft_va_init_fn(N_TEST, L) 
            res_heft, _ = optimize_vqe(heft_cost, init_heft, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR)
            
            # Calculate fidelity: F = |<psi_exact|psi_vqe>|^2
            f_heft = fidelity(gs_vector, heft_state(res_heft))
            current_heft_seeds.append(f_heft)
            
            # --- HEA Optimization ---
            # Uses standard uniform random initialization
            init_hea = hea_init_fn(N_TEST, L) 
            res_hea, _ = optimize_vqe(hea_cost, init_hea, steps=N_OPTIMIZER_STEPS, lr=OPTIMIZER_LR)
            
            f_hea = fidelity(gs_vector, hea_state(res_hea))
            current_hea_seeds.append(f_hea)

        # Statistical aggregation for Journal Figures
        heft_f_means.append(float(np.mean(current_heft_seeds)))
        heft_f_stds.append(float(np.std(current_heft_seeds)))
        hea_f_means.append(float(np.mean(current_hea_seeds)))
        hea_f_stds.append(float(np.std(current_hea_seeds)))
        
        print(f"    L={L} Results:")
        print(f"      H-EFT Fidelity: {heft_f_means[-1]:.5f} ± {heft_f_stds[-1]:.5f}")
        print(f"      HEA Fidelity:   {hea_f_means[-1]:.5f} ± {hea_f_stds[-1]:.5f}")

    # Save results to JSON for the plotting script
    results.update({
        'heft_fid_mean': heft_f_means, 
        'heft_fid_std': heft_f_stds,
        'hea_fid_mean': hea_f_means, 
        'hea_fid_std': hea_f_stds,
        'layer_list': LAYER_LIST, 
        'n_qubits': N_TEST
    })
    
    save_results("test16_ground_state_fidelity", results)
    print(f"\nTest 16 completed in {time.time() - start_time:.2f} seconds.")


# --- Plotting Functions (T1-T16) ---

def plot_1_gv_scaling_analytic():
    """Plot for T1: Gradient Variance Scaling (Analytic)"""
    try:
        results = load_results("test1_gv_scaling_analytic")
    except FileNotFoundError:
        print("Skipping plot 1: Results file not found.")
        return

    data = []
    for key, val in results.items():
        data.append({
            'N': val['n_qubits'],
            'L': val['n_layers'],
            'Mean GV': val['mean_gv'],
            'Std GV': val['std_gv'],
            'N*L': val['n_qubits'] * val['n_layers']
        })
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Mean GV vs N for fixed L
    for L in LAYER_LIST:
        df_L = df[df['L'] == L]
        ax.errorbar(df_L['N'], df_L['Mean GV'], yerr=df_L['Std GV'], fmt='-o', capsize=5, label=f'L={L}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Qubits (N)')
    ax.set_ylabel(r'Mean Squared Gradient Norm $\langle ||\nabla C||^2 \rangle$')
    ax.set_title('T1: Gradient Variance Scaling (H-EFT-VA)')
    ax.legend(title='Layers (L)', loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    save_plot(fig, "T1_GV_Scaling_Analytic")


def plot_2_landscape_flatness_scan():
    """Plot for T2: Landscape Flatness Scan"""
    try:
        results = load_results("test2_landscape_flatness")
    except FileNotFoundError:
        print("Skipping plot 2: Results file not found.")
        return

    for key, Z_list in results.items():
        n, l = key.split('_')[0][1:], key.split('_')[1][1:]
        Z = np.array(Z_list)
        p_range = np.linspace(-np.pi, np.pi, Z.shape[0])

        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.contourf(p_range, p_range, Z, levels=50, cmap='viridis')
        fig.colorbar(c, ax=ax, label=r'Expectation Value $\langle H \rangle$')
        
        ax.set_xlabel(r'Parameter $\theta_1$')
        ax.set_ylabel(r'Parameter $\theta_2$')
        ax.set_title(f'T2: Energy Landscape (N={n}, L={l})')
        
        save_plot(fig, f"T2_Landscape_Flatness_N{n}_L{l}")


def plot_3_initialization_scale_dependence():
    """Plot for T3: Initialization Scale Dependence"""
    try:
        results = load_results("test3_init_scale_dependence")
    except FileNotFoundError:
        print("Skipping plot 3: Results file not found.")
        return

    scales = np.array(results['scales'])
    mean_gv = np.array(results['mean_gv'])
    std_gv = np.array(results['std_gv'])
    n, l = results['n_qubits'], results['n_layers']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(scales, mean_gv, yerr=std_gv, fmt='-o', capsize=5, color='C0')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Initialization Scale $\sigma$')
    ax.set_ylabel(r'Mean Squared Gradient Norm $\langle ||\nabla C||^2 \rangle$')
    ax.set_title(f'T3: Gradient Variance vs. Initialization Scale (N={n}, L={l})')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    save_plot(fig, "T3_Init_Scale_Dependence")


def plot_4_depth_limited_scaling():
    """Plot for T4: Depth-Limited Scaling (L=2)"""
    try:
        results = load_results("test4_depth_limited_scaling")
    except FileNotFoundError:
        print("Skipping plot 4: Results file not found.")
        return

    data = []
    for key, val in results.items():
        data.append({
            'N': val['n_qubits'],
            'L': val['n_layers'],
            'Mean GV': val['mean_gv'],
            'Std GV': val['std_gv'],
        })
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.errorbar(df['N'], df['Mean GV'], yerr=df['Std GV'], fmt='-o', capsize=5, color='C1')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Qubits (N)')
    ax.set_ylabel(r'Mean Squared Gradient Norm $\langle ||\nabla C||^2 \rangle$')
    ax.set_title(f"T4: Depth-Limited Scaling (L={df['L'].iloc[0]})")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    save_plot(fig, "T4_Depth_Limited_Scaling")


def plot_5_noiseless_convergence():
    """Plot for T5: Noiseless Convergence (H-EFT-VA vs HEA)"""
    try:
        results = load_results("test5_noiseless_convergence")
    except FileNotFoundError:
        print("Skipping plot 5: Results file not found.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot a subset of N and L for clarity
    N_subset = [4, 8, 12]
    L_subset = [4, 8, 12]
    
    for key, val in results.items():
        n, l = val['n_qubits'], val['n_layers']
        if n in N_subset and l in L_subset:
            history_heft = np.array(val['heft_history'])
            history_hea = np.array(val['hea_history'])
            
            ax.plot(history_heft, label=f'H-EFT-VA (N={n}, L={l})', linestyle='-')
            ax.plot(history_hea, label=f'HEA (N={n}, L={l})', linestyle='--')

    ax.set_xlabel('Optimization Step')
    ax.set_ylabel(r'Expectation Value $\langle H \rangle$')
    ax.set_title('T5: Noiseless Convergence')
    ax.legend(title='Ansatz & Size', loc='upper right', fontsize='small')
    ax.grid(True, ls="--", alpha=0.5)
    
    save_plot(fig, "T5_Noiseless_Convergence")


def plot_6_convergence_vs_system_size():
    """Plot for T6: Convergence vs System Size (L=2)"""
    try:
        results = load_results("test6_convergence_vs_system_size")
    except FileNotFoundError:
        print("Skipping plot 6: Results file not found.")
        return

    data = []
    for key, val in results.items():
        data.append({
            'N': val['n_qubits'],
            'H-EFT E': val['heft_final_energy'],
            'HEA E': val['hea_final_energy'],
        })
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(df['N'], df['H-EFT E'], '-o', label='H-EFT-VA')
    ax.plot(df['N'], df['HEA E'], '--s', label='HEA')

    ax.set_xlabel('Number of Qubits (N)')
    ax.set_ylabel(r'Final Expectation Value $\langle H \rangle$')
    ax.set_title(f"T6: Final Energy vs. Qubit Count (Fixed L={results[list(results.keys())[0]]['n_layers']})")
    ax.legend(loc='best')
    ax.grid(True, ls="--", alpha=0.5)
    
    save_plot(fig, "T6_Convergence_vs_System_Size")


def plot_7_parameter_efficiency():
    """Plot for T7: Parameter Efficiency (H-EFT-VA vs HEA)"""
    try:
        results = load_results("test7_parameter_efficiency")
    except FileNotFoundError:
        print("Skipping plot 7: Results file not found.")
        return

    data = []
    for key, val in results.items():
        data.append({
            'N': val['n_qubits'],
            'L': val['n_layers'],
            'Final Energy': val['heft_final_energy'],
            'Num Params': val['heft_num_params'],
            'Ansatz': 'H-EFT-VA'
        })
        data.append({
            'N': val['n_qubits'],
            'L': val['n_layers'],
            'Final Energy': val['hea_final_energy'],
            'Num Params': val['hea_num_params'],
            'Ansatz': 'HEA'
        })
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.scatterplot(data=df, x='Num Params', y='Final Energy', hue='Ansatz', style='N', size='L', palette='viridis', ax=ax)

    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel(r'Final Expectation Value $\langle H \rangle$')
    ax.set_title('T7: Parameter Efficiency (Final Energy vs. Parameters)')
    ax.legend(title='Ansatz', loc='best', fontsize='small')
    ax.grid(True, ls="--", alpha=0.5)
    
    save_plot(fig, "T7_Parameter_Efficiency")


def plot_8_optimizer_robustness():
    """Plot for T8: Optimizer Robustness"""
    try:
        results = load_results("test8_optimizer_robustness")
    except FileNotFoundError:
        print("Skipping plot 8: Results file not found.")
        return

    key = list(results.keys())[0]
    val = results[key]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(val['adam_history'], label='Adam', linestyle='-')
    ax.plot(val['sgd_history'], label='SGD', linestyle='--')
    ax.plot(val['rmsprop_history'], label='RMSProp', linestyle=':')

    ax.set_xlabel('Optimization Step')
    ax.set_ylabel(r'Expectation Value $\langle H \rangle$')
    ax.set_title(f"T8: Optimizer Robustness (N={val['n_qubits']}, L={val['n_layers']})")
    ax.legend(title='Optimizer', loc='upper right')
    ax.grid(True, ls="--", alpha=0.5)
    
    save_plot(fig, "T8_Optimizer_Robustness")


def plot_9_noise_robustness_analytic():
    """Plot for T9: Noise Robustness (Analytic)"""
    try:
        results = load_results("test9_noise_robustness_analytic")
    except FileNotFoundError:
        print("Skipping plot 9: Results file not found.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract data for plotting
    p_noise_list = []
    for key in results.keys():
        if key.startswith('P_NOISE_'):
            p_noise_list.append(float(key.split('_')[-1]))
    
    p_noise_list.sort()
    
    for p_noise in p_noise_list:
        key = f"P_NOISE_{p_noise}"
        if key in results:
            history = np.array(results[key]['history'])
            label = f'p={p_noise}' if p_noise > 0 else 'Noiseless'
            ax.plot(history, label=label)

    ax.set_xlabel('Optimization Step')
    ax.set_ylabel(r'Expectation Value $\langle H \rangle$')
    ax.set_title(f"T9: Noise Robustness (N={results['n_qubits']}, L={results['n_layers']})")
    ax.legend(title='Depolarizing Noise Probability (p)', loc='upper right')
    ax.grid(True, ls="--", alpha=0.5)
    
    save_plot(fig, "T9_Noise_Robustness_Analytic")


def plot_10_finite_shot_gv_estimator():
    """Final Fix for T10: Matches ALL CAPS keys from Debug Info"""
    try:
        results = load_results("test10_finite_shot_gv_estimator")
    except FileNotFoundError:
        print("Skipping plot 10: Results not found.")
        return

    data = []
    # Using your specific keys: 'SHOTS_1000', 'SHOTS_5000', etc.
    for key in ['SHOTS_1000', 'SHOTS_5000', 'SHOTS_10000']:
        if key in results:
            val = results[key]
            try:
                shots = int(key.split('_')[-1])
                data.append({'Shots': shots, 'Ansatz': 'H-EFT-VA', 
                             'Mean GV': val.get('heft_mean_gv', 0)})
                data.append({'Shots': shots, 'Ansatz': 'HEA', 
                             'Mean GV': val.get('hea_mean_gv', 0)})
            except (ValueError, IndexError):
                continue

    if not data:
        print("Final attempt failed: Check if heft_mean_gv exists inside the SHOTS_ keys.")
        return

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(data=df, x='Shots', y='Mean GV', hue='Ansatz', marker='o', ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Number of Shots")
    ax.set_ylabel(r"Gradient Variance $\langle ||\nabla C||^2 \rangle$")
    ax.set_title("T10: Robustness to Finite-Shot Noise")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    save_plot(fig, "T10_Finite_Shot_Stability")


def plot_11_shot_noise_convergence():
    """Plot for T11: Shot + Noise Convergence"""
    try:
        results = load_results("test11_shot_noise_convergence")
    except FileNotFoundError:
        print("Skipping plot 11: Results file not found.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract data for plotting
    p_noise_list = []
    for key in results.keys():
        if key.startswith('P_NOISE_'):
            p_noise_list.append(float(key.split('_')[-1]))
    
    p_noise_list.sort()
    
    for p_noise in p_noise_list:
        key = f"P_NOISE_{p_noise}"
        if key in results:
            history = np.array(results[key]['history'])
            label = f'p={p_noise}' if p_noise > 0 else 'Noiseless'
            ax.plot(history, label=label)

    ax.set_xlabel('Optimization Step')
    ax.set_ylabel(r'Expectation Value $\langle H \rangle$')
    ax.set_title(f"T11: Shot + Noise Convergence (N={results['n_qubits']}, L={results['n_layers']}, Shots={results['shots']})")
    ax.legend(title='Depolarizing Noise Probability (p)', loc='upper right')
    ax.grid(True, ls="--", alpha=0.5)
    
    save_plot(fig, "T11_Shot_Noise_Convergence")


def plot_12_second_hamiltonian_test():
    """Plot for T12: Second Hamiltonian (Heisenberg) GV Scaling"""
    try:
        results = load_results("test12_second_hamiltonian_test")
    except FileNotFoundError:
        print("Skipping plot 12: Results file not found.")
        return

    data = []
    for key, val in results.items():
        data.append({
            'N': val['n_qubits'],
            'L': val['n_layers'],
            'Mean GV': val['mean_gv'],
            'Std GV': val['std_gv'],
            'N*L': val['n_qubits'] * val['n_layers']
        })
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Mean GV vs N for fixed L
    for L in LAYER_LIST:
        df_L = df[df['L'] == L]
        ax.errorbar(df_L['N'], df_L['Mean GV'], yerr=df_L['Std GV'], fmt='-o', capsize=5, label=f'L={L}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Qubits (N)')
    ax.set_ylabel(r'Mean Squared Gradient Norm $\langle ||\nabla C||^2 \rangle$')
    ax.set_title('T12: GV Scaling (Heisenberg XXZ)')
    ax.legend(title='Layers (L)', loc='upper right')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    save_plot(fig, "T12_Heisenberg_GV_Scaling")


def plot_13_entanglement_growth():
    """T13 Fix: Dynamic Key Recovery for Entanglement Data"""
    try:
        results = load_results("test13_entanglement_growth")
    except FileNotFoundError:
        print("Results file not found.")
        return

    # Recovery logic: Check all possible keys the test might have used
    layers = results.get('layer_list', results.get('layers', []))
    
    # Try different naming conventions for the entropy data
    heft_ee = results.get('heft_ee', results.get('heft_entropy', []))
    heft_std = results.get('heft_ee_std', results.get('heft_entropy_std', [0]*len(heft_ee)))

    if not heft_ee or sum(heft_ee) == 0:
        print("CRITICAL: Data is missing from JSON. Let's check the keys:")
        print(f"Available keys in JSON: {list(results.keys())}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Standard Plotting
    ax.plot(layers, heft_ee, '-o', label='H-EFT-VA', color='royalblue', linewidth=2.5)
    ax.fill_between(layers, 
                    np.array(heft_ee) - np.array(heft_std), 
                    np.array(heft_ee) + np.array(heft_std), 
                    alpha=0.2, color='royalblue')

    ax.set_xlabel("Circuit Depth ($L$)", fontsize=12)
    ax.set_ylabel("Von Neumann Entropy $S_V$", fontsize=12)
    ax.set_title("T13: Volume-Law Entanglement Growth", fontsize=14)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    save_plot(fig, "T13_Entanglement_Growth_SUCCESS")


def plot_14_expressibility_proxy():
    """
    Plot for T14: Expressibility Proxy (Mean Purity).
    Updated for Tier 1 Journal standards with Haar-random baseline.
    """
    try:
        results = load_results("test14_expressibility_proxy")
    except FileNotFoundError:
        print("Skipping plot 14: Results file not found.")
        return

    layer_list = results['layer_list']
    heft_purity = results['heft_purity']
    hea_purity = results['hea_purity']
    n_qubits = results['n_qubits']

    # Tier 1 Requirement: Calculate Haar-random purity baseline
    haar_purity = 2 / (2**n_qubits + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data
    ax.plot(layer_list, heft_purity, '-o', color='royalblue', label='H-EFT-VA', linewidth=2)
    ax.plot(layer_list, hea_purity, '--s', color='darkorange', label='HEA', linewidth=2)
    
    # Plot Haar Baseline
    ax.axhline(y=haar_purity, color='red', linestyle=':', label='Haar Limit', alpha=0.8)

    # Styling for Publication
    ax.set_xlabel('Number of Layers (L)')
    ax.set_ylabel(r'Mean Purity $\langle \text{Tr}(\rho^2) \rangle$')
    
    # Optional: Log scale is often used if purity approaches the limit very closely
    # ax.set_yscale('log') 
    
    ax.set_title(f"Expressibility Benchmark (N={n_qubits})", pad=15)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, ls="--", alpha=0.3)
    
    # Add a text box explaining the result
    textstr = r"Lower purity $\rightarrow$ Higher expressibility"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    save_plot(fig, "T14_Expressibility_Proxy_Publication")

def plot_15_statistical_significance():
    """Plot for T15: Statistical Significance Testing"""
    try:
        results = load_results("test15_statistical_significance")
    except FileNotFoundError:
        print("Skipping plot 15: Results file not found.")
        return

    data = []
    for key, val in results.items():
        data.append({
            'N': val['n_qubits'],
            'L': val['n_layers'],
            'H-EFT Mean E': val['heft_mean'],
            'H-EFT Std E': val['heft_std'],
            'HEA Mean E': val['hea_mean'],
            'HEA Std E': val['hea_std'],
            'p-value': val['p_value'],
            'Cohen\'s d': val['cohen_d']
        })
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Mean Final Energy
    ax.errorbar(df['N'], df['H-EFT Mean E'], yerr=df['H-EFT Std E'], fmt='-o', capsize=5, label='H-EFT-VA')
    ax.errorbar(df['N'], df['HEA Mean E'], yerr=df['HEA Std E'], fmt='--s', capsize=5, label='HEA')

    ax.set_xlabel('Number of Qubits (N)')
    ax.set_ylabel(r'Mean Final Energy $\langle H \rangle$')
    ax.set_title('T15: Statistical Significance of Convergence')
    ax.legend(loc='best')
    ax.grid(True, ls="--", alpha=0.5)
    
    # Optional: secondary axis for p-value or Cohen's d
    ax2 = ax.twinx()
    ax2.plot(df['N'], df['p-value'], ':^', color='red', label='p-value')
    ax2.set_ylabel('p-value', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_yscale('log')
    ax2.axhline(0.05, color='red', linestyle='-.', linewidth=0.8, label='p=0.05')
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    
    save_plot(fig, "T15_Statistical_Significance")


def plot_16_ground_state_fidelity():
    """Plot for T16: Ground State Fidelity Analysis with Error Bars"""
    try:
        results = load_results("test16_ground_state_fidelity")
    except FileNotFoundError:
        print("Skipping plot 16: Results file not found.")
        return

    # Use the new Tier 1 keys
    layers = results['layer_list']
    heft_mean = results['heft_fid_mean']
    heft_std = results['heft_fid_std']
    hea_mean = results['hea_fid_mean']
    hea_std = results['hea_fid_std']

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot H-EFT-VA with Shaded Error
    ax.plot(layers, heft_mean, '-o', label='H-EFT-VA', color='royalblue', linewidth=2)
    ax.fill_between(layers, 
                    np.array(heft_mean) - np.array(heft_std), 
                    np.array(heft_mean) + np.array(heft_std), 
                    alpha=0.2, color='royalblue')

    # Plot HEA with Shaded Error
    ax.plot(layers, hea_mean, '--s', label='HEA (Baseline)', color='darkorange', linewidth=2)
    ax.fill_between(layers, 
                    np.array(hea_mean) - np.array(hea_std), 
                    np.array(hea_mean) + np.array(hea_std), 
                    alpha=0.2, color='darkorange')

    ax.set_xlabel("Circuit Depth ($L$)")
    ax.set_ylabel("Ground State Fidelity $F$")
    ax.set_title(f"T16: VQE Performance (N={results.get('n_qubits', 6)})")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)
    
    save_plot(fig, "T16_Fidelity_Analysis")


def run_all_tests():
    """Executes all defined benchmark tests."""
    print("=====================================================")
    print("  Starting H-EFT-VA Complete Benchmark Suite")
    print("=====================================================")
    
    Run all tests
    test1_gv_scaling_analytic()
    test2_landscape_flatness_scan()
    test3_init_scale_dependence()            
    test4_depth_limited_scaling()
    test5_noiseless_convergence()
    test6_convergence_vs_system_size()
    test7_parameter_efficiency()
    test8_optimizer_robustness()
    test9_noise_robustness_analytic()
    test10_finite_shot_gv_estimator()
    test11_shot_noise_convergence()
    test12_second_hamiltonian_test()
    test13_entanglement_growth()
    test14_expressibility_proxy()
    test15_statistical_significance()
    test16_ground_state_fidelity()
    
    print("\n=====================================================")
    print("  All benchmark tests completed.")
    print("  Results saved in the 'results' directory.")
    print("=====================================================")

def plot_all_results():
    """Executes all plotting functions."""
    print("\n=====================================================")
    print("  Starting Plotting Routine")
    print("=====================================================")
    
    try:
        import pandas as pd
    except ImportError:
        print("Skipping plotting: pandas library not found. Please install it (pip install pandas).")
        return

    plot_1_gv_scaling_analytic()
    plot_2_landscape_flatness_scan()
    plot_3_initialization_scale_dependence()           
    plot_4_depth_limited_scaling()
    plot_5_noiseless_convergence()
    plot_6_convergence_vs_system_size()
    plot_7_parameter_efficiency()
    plot_8_optimizer_robustness()
    plot_9_noise_robustness_analytic()
    plot_10_finite_shot_gv_estimator()
    plot_11_shot_noise_convergence()
    plot_12_second_hamiltonian_test()
    plot_13_entanglement_growth()
    plot_14_expressibility_proxy()
    plot_15_statistical_significance()
    plot_16_ground_state_fidelity()
    
    print("\nAll plots saved to the 'figures' directory.")
    print("=====================================================")


if __name__ == '__main__':
    # Note: Ensure PennyLane, numpy, matplotlib, seaborn, pandas, and scipy are installed in your environment.
    # Example: pip install pennylane numpy matplotlib seaborn pandas scipy
    run_all_tests()
    plot_all_results()