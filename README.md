# HawkEye: Quantum Entanglement Swapping & Star-Network Simulation Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![Live Platform](https://img.shields.io/badge/Live_Demo-Render-06b6d4?style=flat&logo=render)](https://hawkeye-quantum.onrender.com)

> 🚀 **Live Interactive Dashboard:** [https://hawkeye-quantum.onrender.com](https://hawkeye-quantum.onrender.com)

**HawkEye** is an open-source, high-performance simulation engine and diagnostic web platform engineered to model, analyze, and benchmark multipartite quantum entanglement distribution over star networks. 

The platform supports both parallel (**Factory**) and sequential (**PieceMaker**) entanglement swapping protocols across arbitrary link scales ($L \ge 2$), incorporating exact Kraus channel operator representations for local quantum memory dephasing noise ($\Lambda$).

---

## Architecture Overview

Quantum repeaters and central entanglement routers face memory decoherence when elementary links are generated asynchronously. HawkEye models a central star topology where an $L$-qubit Greenberger-Horne-Zeilinger (GHZ) state is distilled from $L$ independent Bell pairs via non-local Bell state measurements (BSMs) and feed-forward Pauli corrections.

```
                  [Remote Node 1]
                        |
                        | (Link 1)
 [Remote Node L] --- [Central Router] --- [Remote Node 2]
       ...              (BSM Engine)            ...
                        |
                  [Remote Node k]
```

### Supported Swapping Protocols

1. **Factory Protocol (Parallel Swapping):**
   * Link generation attempts across all $L$ arms occur concurrently.
   * Central BSMs are performed in parallel once links establish.
   * Stored qubits experience memory idle times dependent on the arrival time of the final link ($t_{\max} = \max\{r_k\}$).

2. **PieceMaker Protocol (Sequential Swapping):**
   * Links are sequentially coupled to an active register via iterative swapping operations.
   * Intermediate entanglement states are maintained while awaiting downstream link generation.
   * Optimizes physical memory footprint at the central router.

---

## Theoretical Framework

### Quantum Channel & Memory Dephasing Model

For an elementary link $k$ generated at discrete round $r_k$, the waiting time prior to swapping is:
$$w_k = t_{\max} - r_k, \quad \text{where } t_{\max} = \max_{j} \{ r_j \}$$

Idle quantum memory dephasing is modeled via the single-qubit dephasing channel $\mathcal{E}_{\Lambda}^{(w_k)}$ with coherence parameter $\Lambda \in [0, 1]$:

$$\mathcal{E}_{\Lambda}^{(w)}(\rho) = \frac{1 + \Lambda^w}{2}\rho + \frac{1 - \Lambda^w}{2} Z \rho Z$$

The Kraus representation of the channel acting on link $k$ is given by:
$$K_0 = \sqrt{\frac{1 + \Lambda^{w_k}}{2}}\mathbb{I}, \quad K_1 = \sqrt{\frac{1 - \Lambda^{w_k}}{2}}Z$$

### Hybrid Computational Scaling Pipeline

To balance full density matrix tomography with scalable high-$L$ network benchmarks, HawkEye implements a **two-tier computation engine**:

1. **Microscopic Tensor Contraction ($L \le 5$):**
   * Constructs the full density matrix $\rho \in \mathbb{C}^{2^L \times 2^L}$.
   * Explicitly evaluates projectors, Bell measurements, Kraus contractions, and partial traces.
   * Renders full 2D and 3D state tomography (real amplitudes and imaginary coherences).

2. **Exact Channel Decomposition ($L > 5$):**
   * Solves the diagonal and off-diagonal decay analytically to eliminate the $O(2^{2L})$ Hilbert space memory barrier:

     $$\mathcal{F}_L = \langle \text{GHZ}_L | \rho | \text{GHZ}_L \rangle = \frac{1}{2}\left(1 + \prod_{k=1}^{L}\Lambda^{w_k}\right)$$

     $$\text{Tr}(\rho^2) = \frac{1}{2} + \frac{1}{2}\left(\prod_{k=1}^{L}\Lambda^{w_k}\right)^2$$

   * Evaluates exact metrics for $L = 50, 100, 1000$ links in under $1\,\text{ms}$.

---

## Key Platform Features

* **Interactive GUI Dashboard:** Monolithic, modern Web interface built with TailwindCSS and responsive SVG topology visualizers.
* **Quantum State Tomography:** Dual-engine 2D planar heatmaps and WebGL-accelerated 3D surface visualizations via Plotly.js.
* **REST Simulation API:** Fast asynchronous backend powered by FastAPI and Uvicorn.
* **Real-time Diagnostic Telemetry:** Dynamic readout of state fidelity $\mathcal{F}$, purity $\text{Tr}(\rho^2)$, Hilbert dimension, and Hermiticity validation.

---

## Directory Structure

```text
HawkEye/
├── quanta/
│   ├── __init__.py           # Package namespace
│   ├── app.py                # FastAPI server & simulation endpoints
│   ├── index.html            # HawkEye UI dashboard & WebGL tomography
│   ├── prime_rk.py           # Core quantum linear algebra & matrix ops
│   └── protocols.py          # Factory & PieceMaker swapping protocols
├── requirements.txt          # Python dependencies
├── Procfile                  # Cloud deployment runner
└── README.md                 # Project documentation
```

---

## Installation & Local Development

### Prerequisites
* Python 3.10 or higher
* `pip` and `git`

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/HawkEye.git
   cd HawkEye
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the local development server:**
   ```bash
   python -m uvicorn quanta.app:app --reload --port 8000
   ```

5. **Access the platform:**
   Open your browser and navigate to:
   ```text
   [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
   ```

---

## API Documentation

### Simulation Endpoint: `POST /api/simulate`

#### Request Body
```json
{
  "protocol": "factory",
  "link": 3,
  "dephasing": true,
  "Lambda": 0.95,
  "rounds": [1, 4, 2]
}
```

#### Response Format ($L \le 5$ Example)
```json
{
  "status": "success",
  "mode": "full_matrix_tomography",
  "protocol": "factory",
  "link": 3,
  "fidelity": 0.932845,
  "density_matrix": {
    "real": [[0.5, 0.0], [0.0, 0.5]],
    "imag": [[0.0, 0.0], [0.0, 0.0]],
    "dims": [[2, 2, 2], [2, 2, 2]],
    "shape": [8, 8],
    "isherm": true,
    "purity": 0.874598
  }
}
```

---

## Deployment

### Render / Cloud Hosting
To deploy to platforms like [Render](https://render.com):
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `uvicorn quanta.app:app --host 0.0.0.0 --port $PORT`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.