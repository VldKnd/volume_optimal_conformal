import unittest

from configs.calibrators import GlobalOTCPCalibratorConfig
from configs.conformal import ResidualConformalPredictorConfig
from experimentation.runner import _conformal_calibration_settings


class ConformalCheckpointConfigTest(unittest.TestCase):

    def test_volume_evaluation_settings_do_not_affect_compatibility(self) -> None:
        original = ResidualConformalPredictorConfig(
            calibrator=GlobalOTCPCalibratorConfig(seed=3),
            volume_mc_samples=1_000,
            volume_seed=3,
        )
        evaluation = original.model_copy(
            update={
                "volume_mc_samples": 10_000,
                "volume_seed": 17,
            }
        )

        self.assertEqual(
            _conformal_calibration_settings(original),
            _conformal_calibration_settings(evaluation),
        )

    def test_calibration_settings_still_affect_compatibility(self) -> None:
        original = ResidualConformalPredictorConfig(
            calibrator=GlobalOTCPCalibratorConfig(seed=3),
        )
        changed = original.model_copy(update={"coverage_mass": 0.8})

        self.assertNotEqual(
            _conformal_calibration_settings(original),
            _conformal_calibration_settings(changed),
        )


if __name__ == "__main__":
    unittest.main()
