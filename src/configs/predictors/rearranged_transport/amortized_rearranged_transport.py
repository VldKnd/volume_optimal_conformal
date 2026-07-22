from typing import Literal

from configs.predictors.rearranged_transport.rearranged_transport import (
    RearrangedTransportPredictorConfig,
)


class AmortizedRearrangedTransportPredictorConfig(RearrangedTransportPredictorConfig):
    """Configuration for a rearrangement conditioned on coverage mass."""

    type: Literal["amortized_rearranged_transport"] = "amortized_rearranged_transport"


AmortizedRearrangedTransportConfig = AmortizedRearrangedTransportPredictorConfig
