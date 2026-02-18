from dataclasses import dataclass, field
from typing import Optional, Any, Dict


@dataclass
class EngineConfig:
    engine: str                       # "dft" | "mlff" | "mlip" | "pyscf"
    device: str = "cpu"               # "cpu" | "cuda"  (default for non-ML can be cpu)
    bind: str = "127.0.0.1:8888"      # used by mlff/mlip and also pyscf server/client if you do that

    # ML-only (optional)
    model_name: Optional[str] = None
    task_name: Optional[str] = None

    # PySCF-only (optional)
    basis: Optional[str] = None
    functional: Optional[str] = None   # or call this xc

    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingConfig:
    method: Optional[str]
    parameters: dict[str, Any]

@dataclass
class StepContext:
    step_number: int
    operation: str
    engine: str
    charge: int
    multiplicity: int
    engine_cfg: EngineConfig
    sampling: SamplingConfig
    step: dict

@dataclass
class PipelineState:
    last_coords: Any = None
    last_ids: Any = None
    last_energies: Any = None
    last_forces: Any = None

@dataclass
class StepIO:
    step_dir: str
    input_files: list[str]
    output_files: list[str]
    seeds_ids: Optional[list[str]] = None

@dataclass
class StepResult:
    coords: Any
    ids: Any
    energies: Any
    forces: Any
    output_files: list[str]
    
__all__ = [
    "EngineConfig",
    "StepContext",
    "StepResult",
    "PipelineState",
    "SamplingConfig",
]
