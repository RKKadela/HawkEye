import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

# Package imports
from .protocols import Protocol
from .prime_rk import Q, Qp

app = FastAPI(title="HawkEye Quantum Network Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=FileResponse)
def serve_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_path)

class SimulationRequest(BaseModel):
    protocol: str = Field(..., description="'factory' or 'piecemaker'")
    link: int = Field(..., ge=2, le=1000, description="Arbitrary number of links (L >= 2)")
    dephasing: bool = Field(default=False)
    Lambda: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    rounds: Optional[List[int]] = Field(default=None)

    @field_validator("protocol")
    def validate_protocol(cls, v):
        if v.lower() not in ["factory", "piecemaker"]:
            raise ValueError("Protocol must be either 'factory' or 'piecemaker'")
        return v.lower()

    @field_validator("rounds")
    def validate_rounds(cls, v, info):
        link_val = info.data.get("link")
        if v is not None and link_val is not None:
            if len(v) != link_val:
                raise ValueError(f"Length of 'rounds' ({len(v)}) must match 'link' ({link_val})")
            if any(r < 0 for r in v):
                raise ValueError("Round numbers must be non-negative integers")
        return v

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    try:
        sim_rounds = req.rounds if req.rounds is not None else [0] * req.link
        
        # Branch 1: Arbitrary L via Exact Analytical Channel Decomposition
        if req.link > 8:
            if req.dephasing:
                t_max = max(sim_rounds)
                waiting_times = [t_max - r for r in sim_rounds]
                # Coherence damping factor across all links
                coherence_factor = float(np.prod([req.Lambda ** w for w in waiting_times]))
            else:
                coherence_factor = 1.0

            fidelity = 0.5 * (1.0 + coherence_factor)
            purity = 0.5 * (1.0 + (coherence_factor ** 2))

            return {
                "status": "success",
                "mode": "analytical_large_scale",
                "protocol": req.protocol,
                "link": req.link,
                "fidelity": round(fidelity, 6),
                "density_matrix": {
                    "real": None,
                    "imag": None,
                    "dims": [[2] * req.link, [2] * req.link],
                    "shape": [2**req.link, 2**req.link],
                    "isherm": True,
                    "purity": round(purity, 6),
                    "coherence_term": round(0.5 * coherence_factor, 6),
                    "population": 0.5
                }
            }

        # Branch 2: Full State Vector / Density Matrix Tomography (L <= 8)
        if req.protocol == "factory":
            rho = Protocol.factory(
                link=req.link,
                dephasing=req.dephasing,
                Lambda=req.Lambda,
                rounds=sim_rounds,
                verbose=False
            )
        else:
            rho = Protocol.piecemaker(
                link=req.link,
                dephasing=req.dephasing,
                Lambda=req.Lambda,
                rounds=sim_rounds,
                verbose=False
            )

        ideal_target = Protocol.factory(link=req.link, dephasing=False)
        fidelity = float(Qp(rho, ideal_target).fid())
        arr = rho.state

        return {
            "status": "success",
            "mode": "full_matrix_tomography",
            "protocol": req.protocol,
            "link": req.link,
            "fidelity": round(fidelity, 6),
            "density_matrix": {
                "real": np.real(arr).tolist(),
                "imag": np.imag(arr).tolist(),
                "dims": rho.dims,
                "shape": list(arr.shape),
                "isherm": bool(rho.isherm),
                "purity": round(float(np.real(np.trace(arr @ arr)).item()), 6)
            }
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

def clean_matrix_payload(rho: Q):
    """Sanitizes complex quantum density matrix for JSON transport."""
    arr = rho.state
    return {
        "real": np.real(arr).tolist(),
        "imag": np.imag(arr).tolist(),
        "dims": rho.dims,
        "shape": list(arr.shape),
        "isherm": bool(rho.isherm),
        "purity": float(np.real(np.trace(arr @ arr)).item())
    }

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    try:
        # Default generation rounds if omitted
        sim_rounds = req.rounds if req.rounds is not None else [0] * req.link

        if req.protocol == "factory":
            rho = Protocol.factory(
                link=req.link,
                dephasing=req.dephasing,
                Lambda=req.Lambda,
                rounds=sim_rounds,
                verbose=False
            )
        else:
            rho = Protocol.piecemaker(
                link=req.link,
                dephasing=req.dephasing,
                Lambda=req.Lambda,
                rounds=sim_rounds,
                verbose=False
            )

        # Baseline comparison: compute ideal target state fidelity
        ideal_target = Protocol.factory(link=req.link, dephasing=False)
        fidelity = float(Qp(rho, ideal_target).fid())

        return {
            "status": "success",
            "protocol": req.protocol,
            "fidelity": fidelity,
            "density_matrix": clean_matrix_payload(rho)
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))