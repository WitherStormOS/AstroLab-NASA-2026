import unittest

from aerolab import CRITICAL_SINGULARITY, FluidVelocityEngine


class AeroLabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FluidVelocityEngine()

    def test_dynamic_fixture(self) -> None:
        self.assertEqual(
            self.engine.calculate_sector3_status(120, 4, "0.5", 6),
            {"Status": "STABLE", "F_Newton": 18.0, "F_int_Units": 18_000_000},
        )

    def test_manual_reference_fixture(self) -> None:
        self.assertEqual(
            self.engine.calculate_sector3_status(
                80, 2, "0.5", 4, mode="manual_reference"
            ),
            {"Status": "STABLE", "F_Newton": 22.0, "F_int_Units": 22_000_000},
        )

    def test_zero_time_is_critical(self) -> None:
        self.assertEqual(
            self.engine.calculate_sector3(80, 0, "0.5", 4),
            CRITICAL_SINGULARITY,
        )

    def test_does_not_depend_on_astrolab(self) -> None:
        self.assertNotIn("astrolab", FluidVelocityEngine.__module__)


if __name__ == "__main__":
    unittest.main()
