import numpy as np
from .prime_rk import Q, Qp, fold


class Protocol: # Change name to StarProtocol
    """Encapsulates quantum network entanglement swapping protocols."""

    @staticmethod
    def factory(
        link, dephasing=False, Lambda=None, rounds=None, verbose=False
    ):
        """Factory protocol for entanglement swapping in a quantum star network.

        Supports both ideal execution and pure dephasing noise parameterized by
        link generation rounds.
        """
        # ----------------------------------------------------
        # 1. Operators and Initial State Setup
        # ----------------------------------------------------
        N = 2 * link  # total number of qubits
        I = Q.Pauli.I
        X = Q.Pauli.X
        Z = Q.Pauli.Z
        zero = Q.basis(2, 0)
        one = Q.basis(2, 1)

        eta = Q.Bell.phi_plus
        rho_0 = eta.dm()
        rho_link = [
            Q(rho_0.state.copy(), dims=rho_0.dims) for _ in range(link)
        ]

        # ----------------------------------------------------
        # 2. Dephasing Noise (Optional)
        # ----------------------------------------------------
        if dephasing:
            if Lambda is None:
                raise ValueError(
                    "Parameter 'Lambda' is required when dephasing=True."
                )
            if rounds is None:
                rounds = [0] * link
            elif len(rounds) != link:
                raise ValueError(
                    f"Length of 'rounds' ({len(rounds)}) must match 'link' ({link})."
                )

            t_max = max(rounds)
            w = [t_max - r for r in rounds]

            if verbose:
                print("rounds in which links are generated :", rounds)
                print("waiting time for each link :", w)

            K_0 = np.sqrt((1 + Lambda) / 2) * I
            K_1 = np.sqrt((1 - Lambda) / 2) * Z
            K0_I = Qp.tp(K_0, I)
            K1_I = Qp.tp(K_1, I)

            for i in range(link):
                for _ in range(w[i]):
                    rho_link[i] = (
                        K0_I @ rho_link[i] @ K0_I.d()
                        + K1_I @ rho_link[i] @ K1_I.d()
                    )

        # Composite state before gates
        rho = fold(Qp.tp, rho_link)

        # ----------------------------------------------------
        # 3. Entangling CNOT Operations
        # ----------------------------------------------------
        P_00 = Qp(zero, zero).op()
        P_11 = Qp(one, one).op()

        opr1 = [I] * N
        opr1[1] = P_00
        A = fold(Qp.tp, opr1)

        for i in range(2, N, 2):
            if i + 1 < N:
                opr2 = [I] * N
                opr2[1] = P_11
                opr2[i + 1] = X
                cnot_gate = A + fold(Qp.tp, opr2)
                rho = cnot_gate @ rho @ cnot_gate.d()

        # ----------------------------------------------------
        # 4. Spoke Ancilla Measurements & Feedforward X Corrections
        # ----------------------------------------------------
        t = [j + 1 for j in range(2, N, 2) if j + 1 < N]

        for step_idx, qubit_idx in enumerate(t):
            m1 = [I] * N
            m2 = [I] * N
            m1[qubit_idx] = P_00
            m2[qubit_idx] = P_11
            M_0 = fold(Qp.tp, m1)
            M_1 = fold(Qp.tp, m2)

            p_0 = np.real((M_0 @ rho).tr())
            p_1 = np.real((M_1 @ rho).tr())
            p_sum = p_0 + p_1
            prob = [p_0 / p_sum, p_1 / p_sum]

            outcome = np.random.choice(["0", "1"], p=prob)
            if verbose:
                print(f"Measurement outcome of q{qubit_idx} qubit : {outcome}")

            if outcome == "0":
                rho = (M_0 @ rho @ M_0.d()) / prob[0]
            else:
                rho = (M_1 @ rho @ M_1.d()) / prob[1]

            if int(outcome) == 1:
                x_gate_ops = [I] * N
                x_gate_ops[qubit_idx - 1] = X
                X_m = fold(Qp.tp, x_gate_ops)
                rho = X_m @ rho @ X_m.d()

        # ----------------------------------------------------
        # 5. Hub Hadamard, Measurement & Feedforward Z Correction
        # ----------------------------------------------------
        h = Q.Gate.H
        H_ops = [I] * N
        H_ops[1] = h
        H = fold(Qp.tp, H_ops)
        rho = H @ rho @ H.d()

        z1_ops = [I] * N
        z2_ops = [I] * N
        z1_ops[1] = P_00
        z2_ops[1] = P_11
        z1 = fold(Qp.tp, z1_ops)
        z2 = fold(Qp.tp, z2_ops)

        p0_q1 = np.real((z1 @ rho).tr())
        p1_q1 = np.real((z2 @ rho).tr())
        prob_q1 = np.array([p0_q1, p1_q1])
        prob_q1 = prob_q1 / np.sum(prob_q1)

        outcome_q1 = np.random.choice(["0", "1"], p=prob_q1)
        if verbose:
            print("Measurement outcome of q1 qubit :", outcome_q1)

        if outcome_q1 == "0":
            rho = (z1 @ rho @ z1.d()) / prob_q1[0]
        else:
            rho = (z2 @ rho @ z2.d()) / prob_q1[1]

        if int(outcome_q1) == 1:
            z_ops = [I] * N
            z_ops[0] = Z
            Z_gate = fold(Qp.tp, z_ops)
            rho = Z_gate @ rho @ Z_gate.d()

        # ----------------------------------------------------
        # 6. Partial Trace
        # ----------------------------------------------------
        trace_qubits = set([1] + t)
        keep_qubits = [k for k in range(N) if k not in trace_qubits]

        rho_final = rho.ptrace(keep=keep_qubits)
        return rho_final

    @staticmethod
    def piecemaker(
        link, dephasing=False, Lambda=None, rounds=None, verbose=False
    ):
        """Simulates the PieceMaker protocol for quantum entanglement swapping

        using native Q, Qp, Q.SWAP, and fold routines.
        """
        # ----------------------------------------------------
        # 1. Operators and Initial State Setup
        # ----------------------------------------------------
        zero = Q.basis(2, 0)
        one = Q.basis(2, 1)

        I = Q.Pauli.I
        X = Q.Pauli.X
        Z = Q.Pauli.Z
        Y = Q.Pauli.Y

        P_00 = Qp(zero, zero).op()
        P_11 = Qp(one, one).op()
        base = [zero, one]

        CNOT = Q.CNOT
        eta = Q.Bell.phi_plus

        N = 2 * link  # total number of qubits
        rho_0 = eta.dm()
        rho_link = [
            Q(rho_0.state.copy(), dims=rho_0.dims) for _ in range(link)
        ]

        # ----------------------------------------------------
        # 2. Dephasing Noise (Optional)
        # ----------------------------------------------------
        if dephasing:
            if Lambda is None:
                raise ValueError(
                    "Parameter 'Lambda' is required when dephasing=True."
                )
            if rounds is None:
                rounds = [0] * link
            elif len(rounds) != link:
                raise ValueError(
                    f"Length of 'rounds' ({len(rounds)}) must match 'link' ({link})."
                )

            t_max = max(rounds)
            w = [t_max - r for r in rounds]

            if verbose:
                print("rounds in which links are generated :", rounds)
                print("waiting time for each link :", w)

            K_0 = np.sqrt((1 + Lambda) / 2) * I
            K_1 = np.sqrt((1 - Lambda) / 2) * Z
            K0_I = Qp.tp(K_0, I)
            K1_I = Qp.tp(K_1, I)

            for i in range(link):
                for _ in range(w[i]):
                    rho_link[i] = (
                        K0_I @ rho_link[i] @ K0_I.d()
                        + K1_I @ rho_link[i] @ K1_I.d()
                    )

        a = [rho_link[k] for k in range(1, len(rho_link))]

        # ----------------------------------------------------
        # 3. Sequential Swapping & Ancilla Measurements
        # ----------------------------------------------------
        rho = rho_link[0]
        d = 0
        e = 2

        for i in range(N - 1):
            dim = 2 * (i + 2) - d
            if dim <= N - d:
                rho = Qp.tp(rho, a[i])

                # SWAP gate using general SWAP
                P = Q.SWAP(dim, dim - 3, dim - 2)
                rho = P @ rho @ P.d()

                # CNOT gate
                c = [I] * (dim - 1)
                c[-1] = CNOT
                Cx = fold(Qp.tp, c)
                rho = Cx @ rho @ Cx.d()

                # Measurement operators
                m_0 = [I] * dim
                m_0[-1] = P_00
                M_0 = fold(Qp.tp, m_0)

                m_1 = [I] * dim
                m_1[-1] = P_11
                M_1 = fold(Qp.tp, m_1)

                p_0 = np.real((M_0 @ rho).tr())
                p_1 = np.real((M_1 @ rho).tr())
                p_sum = p_0 + p_1
                prob = [p_0 / p_sum, p_1 / p_sum]

                outcome = np.random.choice(["0", "1"], p=prob)
                if verbose:
                    print(
                        f"Measurement outcome of qubit (step {i + 1}) : {outcome}"
                    )

                if outcome == "0":
                    rho = (M_0 @ rho @ M_0.d()) / prob[0]
                else:
                    rho = (M_1 @ rho @ M_1.d()) / prob[1]

                # Feedforward X gate
                b_val = int(outcome)
                if b_val == 1:
                    X_m = [I] * dim
                    X_m[-3] = X
                    X_m = fold(Qp.tp, X_m)
                    rho = X_m @ rho @ X_m.d()

                # Subspace projection (exclude measured qubit and correct dims)
                M = [I] * dim
                M[-1] = base[b_val].d()
                M = fold(Qp.tp, M)
                M.dims = [[2] * (dim - 1), [2] * dim]
                rho = M @ rho @ M.d()

                d = i + 1
                e = dim - 1

        # ----------------------------------------------------
        # 4. Central Hub Hadamard & Measurement
        # ----------------------------------------------------
        H_gate_single = Q.Gate.H
        H = [I] * e
        H[-1] = H_gate_single
        H_gate = fold(Qp.tp, H)
        rho = H_gate @ rho @ H_gate.d()

        m1 = [I] * e
        m1[-1] = P_00
        M_0_q2 = fold(Qp.tp, m1)

        m2 = [I] * e
        m2[-1] = P_11
        M_1_q2 = fold(Qp.tp, m2)

        p_0 = np.real((M_0_q2 @ rho).tr())
        p_1 = np.real((M_1_q2 @ rho).tr())
        p_sum = p_0 + p_1
        prob_q2 = [p_0 / p_sum, p_1 / p_sum]

        outcome_q2 = np.random.choice(["0", "1"], p=prob_q2)
        if verbose:
            print("Measurement outcome of q2 qubit :", outcome_q2)

        if outcome_q2 == "0":
            rho = (M_0_q2 @ rho @ M_0_q2.d()) / prob_q2[0]
        else:
            rho = (M_1_q2 @ rho @ M_1_q2.d()) / prob_q2[1]

        # Feedforward Z gate on 1st qubit
        b2 = int(outcome_q2)
        if b2 == 1:
            Z1_m = [I] * e
            Z1_m[0] = Z
            Z1 = fold(Qp.tp, Z1_m)
            rho = Z1 @ rho @ Z1.d()

        # Subspace projection (exclude measured central qubit and correct dims)
        M_final = [I] * e
        M_final[-1] = base[b2].d()
        M_final = fold(Qp.tp, M_final)
        M_final.dims = [[2] * (e - 1), [2] * e]
        rho = M_final @ rho @ M_final.d()

        return rho




# Convenience top-level instances / functions:
protocol = Protocol()
factory_protocol = Protocol.factory
piecemaker_protocol = Protocol.piecemaker
