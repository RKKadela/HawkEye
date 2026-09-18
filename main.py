# import numpy as np
# from quanta import *
# #from quanta.piecemaker import piecemaker_protocol
# #from quanta.factory import factory_protocol
# from quanta.protocols import *
from quanta.prime_rk import Q, Qp
from quanta.protocols import Protocol

# Expose convenience references for interactive terminal testing
factory_protocol = Protocol.factory
piecemaker_protocol = Protocol.piecemaker

if __name__ == "__main__":
    print("--- HawkEye Quantum Network Console ---")
    print("Simulating ideal 3-link star network...")
    
    rho_f = factory_protocol(link=3, verbose=False)
    rho_p = piecemaker_protocol(link=3, verbose=False)
    
    fid = Qp(rho_f, rho_p).fid()
    print(f"Factory vs PieceMaker Output Fidelity: {fid}")