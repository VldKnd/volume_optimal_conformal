from typing import Literal

from pydantic import BaseModel, Field


class GlobalOTCPCalibratorConfig(BaseModel):
    type: Literal["global_otcp"] = "global_otcp"
    seed: int = 0


class LocalOTCPCalibratorConfig(BaseModel):
    type: Literal["local_otcp"] = "local_otcp"
    n_neighbors: int = Field(default=100, gt=1)
    seed: int = 0
