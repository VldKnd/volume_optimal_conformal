from typing import Literal

from configs.trainers.rearranged_transport.amortized_rearranged_transport import (
    AmortizedRearrangedTransportTrainerConfig,
)


class ExperimentalAmortizedRearrangementTrainerConfig(
    AmortizedRearrangedTransportTrainerConfig
):
    """Configuration for direct mean-log amortized rearrangement training."""

    type: Literal["experimental_amortized_rearrangement"] = (
        "experimental_amortized_rearrangement"
    )
