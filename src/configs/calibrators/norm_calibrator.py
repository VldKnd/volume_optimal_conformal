from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class NormCalibratorConfig(BaseModel):
    type: Literal["norm"] = "norm"
    p: float = Field(
        default=2.0,
        gt=0.0,
        validation_alias=AliasChoices("p", "norm"),
    )
