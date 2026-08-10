from typing import Literal

from pydantic import BaseModel, Field


class CDFCalibratorConfig(BaseModel):
    """Configuration for Monte Carlo conditional-density rank scores."""

    type: Literal["cdf_calibrator", "cdf"] = "cdf_calibrator"
    n_cdf_samples: int = Field(default=1_000, gt=0)
    cdf_batch_size: int = Field(
        default=65_536,
        gt=0,
        description=(
            "Approximate maximum number of flattened conditional samples "
            "evaluated in one predictor call."
        ),
    )
