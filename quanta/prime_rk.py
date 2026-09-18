import numpy as np
from scipy.linalg import sqrtm, eigvals


class Q:  # creating objects for quantum states and operators

    def __init__(self, data, dims=None):
        # __init__ method initializes the Q object with a given state or operator.
        # It converts the input data into a NumPy array of complex numbers and stores it in the `state` attribute.
        arr = np.array(data, dtype=complex)

        # Standardize 1D kets into explicit column vectors (N, 1)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        self.state = arr

        # Track Hilbert space dimensions (QuTiP-style)
        if dims is not None:
            self.dims = dims
        else:
            r, c = self.state.shape
            if c == 1:  # Ket vector
                self.dims = [[r], [1]]
            elif r == 1:  # Bra vector
                self.dims = [[1], [c]]
            else:  # Square matrix / general operator
                self.dims = [[r], [c]]

    def __getitem__(self, key):
        """Allows direct indexing into the underlying state array: q[0, 0] or q[0]."""
        return self.state[key]

    def __setitem__(self, key, value):
        """Allows direct in-place modification of underlying elements: q[0, 0] = x."""
        self.state[key] = value

    def copy(self):
            """Returns an independent deep copy of the quantum object."""
            return Q(self.data.copy(), dims=[list(d) for d in self.dims])

    def __array__(self, dtype=None, copy=None):
        """Let Q instances be consumed anywhere a NumPy array is expected

        (np.kron, `@`, np.array(), etc.) without callers needing to unwrap
        `.state` manually.
        """
        arr = self.state
        return arr if dtype is None else arr.astype(dtype)


    def __repr__(self):
        # Show clean floats without `+0.j` when imaginary parts are zero
        data_to_show = (
            self.state.real if np.all(self.state.imag == 0) else self.state
        )

        header = (
            f"Quantum object: dims = {self.dims}, "
            f"shape = {self.state.shape}, "
            f"type = {self.type}\n"
            f"isherm = {self.isherm}\n"
            f"Qobj data =\n{data_to_show}"
        )
        return header
    
    # ==================== QuTiP-style Properties ====================

    @property
    def data(self):
        """Matrix representing the state or operator (NumPy array)."""
        return self.state

    @property
    def shape(self):
        """Dimensions of the underlying data matrix (rows, cols)."""
        return self.state.shape

    @property
    def isherm(self):
        """Check whether the object is a Hermitian square operator (A == A†)."""
        r, c = self.state.shape
        if r != c:
            return False
        return np.allclose(self.state, self.state.conj().T, atol=1e-10)

    @property
    def type(self):
        """Type of the quantum object: 'ket', 'bra', 'oper', or 'other'."""
        r, c = self.state.shape
        if c == 1:
            return "ket"
        elif r == 1:
            return "bra"
        elif r == c:
            return "oper"
        return "other"

    # ----- Basis State Generator (QuTiP-style) -----
    @staticmethod
    def basis(dim, index=0):
        """Generates computational basis ket |index> of dimension dim.

        Usage: Q.basis(2, 0) for |0>, Q.basis(2, 1) for |1>
        """
        if not (0 <= index < dim):
            raise ValueError(
                f"Index {index} out of range for dimension {dim}."
            )
        vec = np.zeros((dim, 1), dtype=complex)
        vec[index, 0] = 1.0
        return Q(vec, dims=[[dim], [1]])

    # ----- Properties / operations -----

    def __add__(self, other):  # Addition (+)
        if isinstance(other, Q):
            return Q(self.state + other.state, dims=self.dims)
        return Q(self.state + other, dims=self.dims)

    def __sub__(self, other):  # Subtraction (-)
        if isinstance(other, Q):
            return Q(self.state - other.state, dims=self.dims)
        return Q(self.state - other, dims=self.dims)

    def __mul__(self, other):  # Scalar Multiplication (*)
        if np.isscalar(other):
            return Q(self.state * other, dims=self.dims)
        raise TypeError(
            "Use '@' for matrix multiplication. '*' is reserved for scalar multiplication."
        )

    def __rmul__(self, other):  # Scalar Multiplication from the Left
        return self.__mul__(other)

    def __truediv__(self, other):  # Scalar Division (/)
        if np.isscalar(other):
            if other == 0:
                raise ZeroDivisionError("Division by zero.")
            return Q(self.state / other, dims=self.dims)
        raise TypeError("Division is only defined by a scalar.")

    # Forces NumPy arrays on the left-hand side of operators (+, -, @) 
    # to yield control to Q methods rather than downcasting to ndarray
    __array_priority__ = 100.0

    # ... [rest of your __init__ and existing code] ...

    def __matmul__(self, other):  # Matrix Multiplication from the Left (self @ other)
        other_state = other.state if isinstance(other, Q) else other
        res = self.state @ other_state
        
        new_dims = None
        if isinstance(other, Q):
            new_dims = [list(self.dims[0]), list(other.dims[1])]
        else:
            # If multiplied by a raw array, preserve self's output dimension
            new_dims = [list(self.dims[0]), [other.shape[1]]]
        return Q(res, dims=new_dims)

    def __rmatmul__(self, other):  # Reflected Matrix Multiplication (other @ self)
        other_state = other.state if isinstance(other, Q) else other
        res = other_state @ self.state
        
        # Preserve input dimensions from self for the right subsystem
        new_dims = [[other.shape[0]], list(self.dims[1])]
        return Q(res, dims=new_dims)

    # def __matmul__(self, other):  # Matrix Multiplication (@)
    #     other_state = other.state if isinstance(other, Q) else other
    #     res = self.state @ other_state
    #     # Derive resulting dimensions: [left_output, right_input]
    #     new_dims = None
    #     if isinstance(other, Q):
    #         new_dims = [list(self.dims[0]), list(other.dims[1])]
    #     return Q(res, dims=new_dims)

    def item(self):
        """Return scalar value if quantum object contains a single element."""
        return self.state.item()

    def d(self):  # dagger
        """Return Hermitian conjugate, wrapped as a Q instance."""
        return Q(self.state.conj().T, dims=[self.dims[1], self.dims[0]])

    def n(self):  # norm: ∥x∥=\sqrt{x^† x} ; \sqrt{Tr(A†A)}
        norm = np.linalg.norm(self.state)
        return round(norm, 4)

    def u(self):  # unit |ψ⟩/∥ψ∥
        """Return normalized state, wrapped as a Q instance."""
        norm = np.linalg.norm(self.state)
        if norm == 0:
            raise ValueError("Cannot normalize zero state/operator.")
        return Q(self.state / norm, dims=self.dims)

    def purity(self):
        """Compute the purity of the state (terminal op -> float)."""
        rows, cols = self.state.shape
        if cols == 1:
            return 1.0
        else:
            result = np.real(np.trace(self.state @ self.state))
            return round(result, 4)

    def dm(self):  # density matrix
        """Return density matrix for a pure state (ket), wrapped as a Q instance."""
        if self.state.ndim != 2 or self.state.shape[1] != 1:
            raise ValueError(
                "Density matrix can only be computed for pure states (column vectors)."
            )
        return Q(
            self.state @ self.state.conj().T, dims=[self.dims[0], self.dims[0]]
        )

    def tr(self):
        """Compute the matrix trace Tr(A) of the quantum object.

        Returns a complex scalar (or float if purely real).
        """
        r, c = self.state.shape
        if r != c:
            raise ValueError("Trace is only defined for square operators.")
        trace_val = np.trace(self.state)
        # Return clean float if imaginary part is negligible
        return (
            float(np.real(trace_val))
            if np.isclose(np.imag(trace_val), 0, atol=1e-10)
            else complex(trace_val)
        )

    def ptrace(self, keep):
        """Partial trace keeping the subsystem indices specified in `keep`.

        Supports composite systems with tracked dimensions.
        """
        if isinstance(keep, int):
            keep = [keep]

        dims_in = list(self.dims[0])
        n_subs = len(dims_in)

        # Reshape into tensor form: (in_0, in_1, ..., out_0, out_1, ...)
        tensor_state = self.state.reshape(dims_in + dims_in)

        # Find which subsystems to trace out
        trace_axes = [i for i in range(n_subs) if i not in keep]

        # Trace over pairs of axes (offset by n_subs for ket/bra parts)
        offset = 0
        for ax in sorted(trace_axes):
            tensor_state = np.trace(
                tensor_state, axis1=ax - offset, axis2=ax + n_subs - 2 * offset
            )
            offset += 1

        new_dims = [dims_in[i] for i in keep]
        new_dim_total = int(np.prod(new_dims))
        return Q(
            tensor_state.reshape(new_dim_total, new_dim_total),
            dims=[new_dims, new_dims],
        )

    # ----- Random state generators -----
    class rand:

        @staticmethod
        def haar(d):
            A = np.random.normal(size=(d, d))
            B = np.random.normal(size=(d, d))
            Z = A + 1j * B
            q_mat, r_mat = np.linalg.qr(Z)
            Lambda = np.diag(
                [r_mat[i, i] / np.abs(r_mat[i, i]) for i in range(d)]
            )
            return Q(q_mat @ Lambda)

        @staticmethod
        def p(n):  # pure state
            d = 2**n
            A = np.random.normal(0, 1 / np.sqrt(2), d)
            B = np.random.normal(0, 1 / np.sqrt(2), d)
            Z = A + 1j * B
            psi = Z / np.linalg.norm(Z)
            dims = [[2] * n, [1] * n]
            return Q(psi.reshape(-1, 1), dims=dims)

        @staticmethod
        def m(n):  # mixed state
            d = 2**n
            A = np.random.normal(0, 1 / np.sqrt(2), (d, d))
            B = np.random.normal(0, 1 / np.sqrt(2), (d, d))
            Z = A + 1j * B
            rho = Z @ Z.conj().T
            rho /= np.trace(rho)
            dims = [[2] * n, [2] * n]
            return Q(rho, dims=dims)

    # ----- Bell states -----
    class Bell:
        phi_plus = np.array([[1], [0], [0], [1]], dtype=complex) / np.sqrt(2)
        phi_minus = np.array([[1], [0], [0], [-1]], dtype=complex) / np.sqrt(2)
        psi_plus = np.array([[0], [1], [1], [0]], dtype=complex) / np.sqrt(2)
        psi_minus = np.array([[0], [1], [-1], [0]], dtype=complex) / np.sqrt(2)

        @classmethod
        def all(cls):
            return {
                "phi_plus": Q(cls.phi_plus, dims=[[2, 2], [1, 1]]),
                "phi_minus": Q(cls.phi_minus, dims=[[2, 2], [1, 1]]),
                "psi_plus": Q(cls.psi_plus, dims=[[2, 2], [1, 1]]),
                "psi_minus": Q(cls.psi_minus, dims=[[2, 2], [1, 1]]),
            }

    class Pauli:
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        @classmethod
        def all(cls):
            return {
                "PauliI": Q(cls.I, dims=[[2], [2]]),
                "PauliX": Q(cls.X, dims=[[2], [2]]),
                "PauliY": Q(cls.Y, dims=[[2], [2]]),
                "PauliZ": Q(cls.Z, dims=[[2], [2]]),
            }

    # ----- Standard Single-Qubit Gates -----
    class Gate:
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    class Werner:

        def __new__(cls, p=None, state=None):
            if state is None:
                state = Q.Bell.phi_plus
            state_arr = state.state if isinstance(state, Q) else state
            bell_pure = state_arr @ state_arr.conj().T
            I4 = np.eye(4) / 4
            if p is None:
                p = np.random.uniform(low=1 / 2, high=1.0)
            return Q(p * bell_pure + (1 - p) * I4, dims=[[2, 2], [2, 2]])

    # ============================================================
    # General controlled single-qubit gate
    # ============================================================
    class ControlGate:

        @staticmethod
        def Cg(gate, n_qubits, control, target):
            if control == target:
                raise ValueError(
                    "Control and target qubits must be different."
                )

            if not (0 <= control < n_qubits):
                raise ValueError("Control qubit index out of range.")

            if not (0 <= target < n_qubits):
                raise ValueError("Target qubit index out of range.")

            gate_arr = gate.state if isinstance(gate, Q) else gate
            dim = 2**n_qubits
            result = np.zeros((dim, dim), dtype=complex)

            for basis_idx in range(dim):
                bits = [
                    (basis_idx >> (n_qubits - 1 - i)) & 1
                    for i in range(n_qubits)
                ]
                if bits[control] == 0:
                    result[basis_idx, basis_idx] = 1
                else:
                    for t in range(2):
                        new_bits = bits.copy()
                        new_bits[target] = t
                        new_basis = 0
                        for b in new_bits:
                            new_basis = (new_basis << 1) | b
                        result[new_basis, basis_idx] = gate_arr[
                            t, bits[target]
                        ]

            return Q(result, dims=[[2] * n_qubits, [2] * n_qubits])

        @staticmethod
        def Cx(n_qubits, control, target):
            return Q.ControlGate.Cg(Q.Pauli.X, n_qubits, control, target)

        @staticmethod
        def Cy(n_qubits, control, target):
            return Q.ControlGate.Cg(Q.Pauli.Y, n_qubits, control, target)

        @staticmethod
        def Cz(n_qubits, control, target):
            return Q.ControlGate.Cg(Q.Pauli.Z, n_qubits, control, target)

    # ============================================================
    # General SWAP gate
    # ============================================================
    @staticmethod
    def SWAP(n_qubits, qubit1, qubit2):
        if qubit1 == qubit2:
            raise ValueError("The two qubits must be different.")

        if not (0 <= qubit1 < n_qubits):
            raise ValueError("qubit1 index out of range.")

        if not (0 <= qubit2 < n_qubits):
            raise ValueError("qubit2 index out of range.")

        dim = 2**n_qubits
        swap = np.zeros((dim, dim), dtype=complex)

        for basis_idx in range(dim):
            bits = [
                (basis_idx >> (n_qubits - 1 - i)) & 1 for i in range(n_qubits)
            ]
            swapped_bits = bits.copy()
            swapped_bits[qubit1], swapped_bits[qubit2] = (
                swapped_bits[qubit2],
                swapped_bits[qubit1],
            )

            new_basis = 0
            for bit in swapped_bits:
                new_basis = (new_basis << 1) | bit
            swap[new_basis, basis_idx] = 1

        return Q(swap, dims=[[2] * n_qubits, [2] * n_qubits])

    # ============================================================
    # Reusable projectors and CNOT gate
    # ============================================================
    proj_0 = (Pauli.I + Pauli.Z) / 2  # |0><0|
    proj_1 = (Pauli.I - Pauli.Z) / 2  # |1><1|
    CNOT = np.kron(proj_0, Pauli.I) + np.kron(proj_1, Pauli.X)

# ============================================================
# Post-Class Initialization: Wrap Static Gates & Operators in Q
# ============================================================

# 1. Pauli Operators
Q.Pauli.I = Q(Q.Pauli.I, dims=[[2], [2]])
Q.Pauli.X = Q(Q.Pauli.X, dims=[[2], [2]])
Q.Pauli.Y = Q(Q.Pauli.Y, dims=[[2], [2]])
Q.Pauli.Z = Q(Q.Pauli.Z, dims=[[2], [2]])

# 2. Standard Single-Qubit Gates
Q.Gate.H = Q(Q.Gate.H, dims=[[2], [2]])
Q.Gate.S = Q(Q.Gate.S, dims=[[2], [2]])
Q.Gate.T = Q(Q.Gate.T, dims=[[2], [2]])

# 3. Canonical Bell States
Q.Bell.phi_plus = Q(Q.Bell.phi_plus, dims=[[2, 2], [1, 1]])
Q.Bell.phi_minus = Q(Q.Bell.phi_minus, dims=[[2, 2], [1, 1]])
Q.Bell.psi_plus = Q(Q.Bell.psi_plus, dims=[[2, 2], [1, 1]])
Q.Bell.psi_minus = Q(Q.Bell.psi_minus, dims=[[2, 2], [1, 1]])

# 4. Projectors & Canonical CNOT
Q.proj_0 = Q(Q.proj_0, dims=[[2], [2]])
Q.proj_1 = Q(Q.proj_1, dims=[[2], [2]])
Q.CNOT = Q(Q.CNOT, dims=[[2, 2], [2, 2]])


class Is:

    def __init__(self, state, tol=1e-10, auto_normalize=True):
        self.state = (
            state.state
            if isinstance(state, Q)
            else np.array(state, dtype=complex)
        )
        self.tol = tol
        self.rows, self.cols = self.state.shape

    def ket(self):
        """Check if the state is a column vector (n×1)."""
        return self.cols == 1

    def square(self):
        """Check if the state is square (n×n)."""
        return self.rows == self.cols

    def herm(self):
        """Check if the state is Hermitian: ρ = ρ†."""
        if not self.square():
            raise ValueError("Hermiticity is defined only for square matrices.")
        return np.allclose(self.state, self.state.conj().T, atol=self.tol)

    def pure(self):
        """Compute the purity of the state."""
        if self.ket():
            return True
        elif self.square():
            if not self.herm():
                raise ValueError(
                    "Density matrix must be Hermitian to test purity."
                )
            purity = np.real(np.trace(self.state @ self.state))
            return abs(purity - 1.0) < self.tol
        else:
            raise ValueError(
                "Given quantum state is neither ket vector nor density matrix."
            )


class Qp:

    def __init__(self, state1, state2=None, tol=1e-10):
        self.state1 = (
            state1.state
            if isinstance(state1, Q)
            else np.array(state1, dtype=complex)
        )
        self.state2 = (
            (
                state2.state
                if isinstance(state2, Q)
                else np.array(state2, dtype=complex)
            )
            if state2 is not None
            else self.state1
        )
        self.tol = tol

    def ip(self):
        if Is(self.state1).ket() and Is(self.state2).ket():
            return (self.state1.conj().T @ self.state2).item()
        elif Is(self.state1).square() and Is(self.state2).square():
            return np.trace(self.state1.conj().T @ self.state2).item()
        else:
            raise ValueError(
                "Inputs must both be vectors (1D) or matrices (2D)."
            )

    def op(self):  # Outer product
        if Is(self.state1).square() or Is(self.state2).square():
            raise ValueError("Both inputs must be 1D vectors.")
        return Q(self.state1 @ self.state2.conj().T)

    @staticmethod
    def tp(*args):  # Tensor product (Arbitrary arguments or sequence)
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]

        if len(args) == 0:
            raise ValueError("tp requires at least one argument.")

        q_objs = [item if isinstance(item, Q) else Q(item) for item in args]

        # Combine underlying matrix states using fold
        combined_matrix = fold(
            np.kron, [q.state.copy() for q in q_objs]
        )  #

        # Combine subsystem dimensions
        dims_0 = []
        dims_1 = []
        for q in q_objs:
            dims_0.extend(q.dims[0])
            dims_1.extend(q.dims[1])

        return Q(combined_matrix, dims=[dims_0, dims_1])

    def fid(self):  # Fidelity
        if Is(self.state1).ket() and Is(self.state2).ket():
            value = self.ip()
            fidelity = np.abs(value).item()
            return round(fidelity, 4)

        elif Is(self.state1).square() and Is(self.state2).square():
            sqrt_rho = sqrtm(self.state1)
            inner = sqrt_rho @ self.state2 @ sqrt_rho
            sqrt_inner = sqrtm(inner)
            fidelity = np.real(np.trace(sqrt_inner)) ** 2
            return round(fidelity, 4)

        elif Is(self.state1).ket() and Is(self.state2).square():
            fidelity = np.abs(
                (Q(self.state1).d() @ self.state2 @ self.state1).item()
            )
            fidelity = np.sqrt(fidelity)
            return round(fidelity, 4)
        elif Is(self.state1).square() and Is(self.state2).ket():
            fidelity = np.abs(
                (Q(self.state2).d() @ self.state1 @ self.state2).item()
            )
            fidelity = np.sqrt(fidelity)
            return round(fidelity, 4)
        else:
            raise ValueError("Unsupported input types for fidelity.")


def conc_pure(state):
    a, b, c, d = state.flatten()
    val = np.abs(np.real(a * d - b * c))
    return round(val, 4)


def conc_mixed(state):
    sy_sy = np.kron(Q.Pauli.Y, Q.Pauli.Y)
    rho_star = np.conj(state)
    rho_tilde = sy_sy @ rho_star @ sy_sy
    R = sqrtm(sqrtm(state) @ rho_tilde @ sqrtm(state))
    eigenvals = np.sort(np.real(eigvals(R)))[::-1]
    concurrence_val = max(0, eigenvals[0] - np.sum(eigenvals[1:]))
    return round(concurrence_val, 4)


class Ebit:

    def __new__(cls, state, method="concurrence"):
        state = (
            state.state
            if isinstance(state, Q)
            else np.asarray(state, dtype=complex)
        )
        method = method.lower() if isinstance(method, str) else "concurrence"

        if state.ndim == 1 or (state.ndim == 2 and state.shape[1] == 1):
            if state.size != 4:
                raise ValueError("Ket must be a 4-dim vector.")
            if method.lower() != "concurrence":
                raise ValueError(
                    f"Method '{method}' not available for pure state."
                )
            return conc_pure(state)

        elif state.shape == (4, 4):
            if method == "concurrence":
                return conc_mixed(state)
            else:
                raise ValueError(f"Unknown entanglement method: {method}")
        else:
            raise ValueError("Input must be 4x1 ket or 4x4 density matrix.")


# fold function for iterative operation reduction
def fold(operation, objects):
    """operation : callable (f(acc, item) -> acc)
    objects   : list or tuple of elements to reduce
    """
    result = objects[0]
    for obj in objects[1:]:
        result = operation(result, obj)
    return result
