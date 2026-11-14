import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from astrolab import (
    CRITICAL_SINGULARITY,
    FluidVelocityEngine,
)
from astrolab_simulations import (
    EARTH_RADIUS_KM,
    simulate_earth_orbit,
)
from astrolab_i18n import TRANSLATIONS, translate
from astrolab_fire import assess_fire_safety, stress_test_measurements
from astrolab_nasa import (
    hitung_simulasi_dataset_contoh,
    hitung_simulasi_integrasi_nasa,
)
from astrolab_visuals import (
    earth_sphere_mesh,
    load_earth_texture,
    render_earth_orthographic,
    visibility_mask,
)


class LocalizationTests(unittest.TestCase):
    def test_ui_strings_have_indonesian_english_and_spanish_options(self) -> None:
        self.assertEqual(set(TRANSLATIONS), {"id", "en", "es"})
        self.assertEqual(translate("Beranda", "id"), "Beranda")
        self.assertEqual(translate("Beranda", "en"), "Home")
        self.assertEqual(translate("Beranda", "es"), "Inicio")
        self.assertEqual(translate("Fire Safety", "en"), "Fire safety")

    def test_missing_translation_falls_back_to_source_text(self) -> None:
        self.assertEqual(translate("Custom text", "es"), "Custom text")


class NasaCsvIntegrationTests(unittest.TestCase):
    def write_csv(self, directory: str, content: str) -> Path:
        path = Path(directory) / "measurements.csv"
        path.write_text(content, encoding="utf-8")
        return path

    def test_legacy_csv_formula_is_calculated_without_reference_override(self) -> None:
        with TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                "test_id,flame_diameter_mm,burn_duration_sec,"
                "airflow_speed_cms,oxygen_concentration\n"
                "BASS_015,26.83,4.0,25.4,21\n",
            )
            result = hitung_simulasi_integrasi_nasa(path).iloc[0]

        self.assertAlmostEqual(result["DA2_units"], 26.83**2 / 3)
        self.assertAlmostEqual(result["Cd"], 0.6)
        self.assertAlmostEqual(result["F_Newton"], 14.997481666666667)
        self.assertEqual(result["F_int_Units"], 14_997_482)
        self.assertAlmostEqual(result["F_formula_Newton"], 14.997481666666667)
        self.assertIsNone(result["F_reference_Newton"])
        self.assertIsNone(result["F_reference_int_Units"])
        self.assertEqual(result["output_kind"], "formula")
        self.assertEqual(result["data_kind"], "legacy")
        self.assertEqual(result["Status"], "STABLE")
        self.assertEqual(result["rho_weight_relative_to_21pct"], 1.0)

    def test_builtin_example_uses_provided_sensor_time_series(self) -> None:
        results = hitung_simulasi_dataset_contoh().set_index("test_id")

        self.assertEqual(len(results), 10)
        sample = results.loc["PSI-25/BASS2_PMMA_RUN04@10.0s"]
        self.assertEqual(sample["INVESTIGATION_ID"], "PSI-25")
        self.assertEqual(sample["TEST_RUN_CODE"], "BASS2_PMMA_RUN04")
        self.assertEqual(sample["TIME_SEC"], 10)
        self.assertEqual(sample["burn_duration_sec"], 10)
        self.assertEqual(sample["DA2_units"], 240)
        self.assertTrue(sample["DA2_is_scenario_assumption"])
        self.assertAlmostEqual(sample["F_Newton"], 32.539370078740156)
        self.assertEqual(sample["fire_status"], "WATCH")
        self.assertIn("not independently verified", sample["data_provenance"])
        self.assertEqual(
            results.loc["PSI-99/SAF2_SIBAL_TEST01@20.0s", "fire_status"],
            "STABLE",
        )
        self.assertEqual(
            results.loc["PSI-25/BASS2_PMMA_RUN04@0.0s", "Status"],
            "NO_PREVIOUS_SAMPLE",
        )

    def test_user_provided_sensor_csv_import_uses_time_series_parser(self) -> None:
        dataset_path = Path(__file__).parent / "data" / "flame_freefall_timeseries.csv"
        result = hitung_simulasi_integrasi_nasa(dataset_path)

        self.assertEqual(len(result), 10)
        self.assertTrue(result["data_kind"].eq("timeseries").all())
        self.assertEqual(
            result.loc[
                result["test_id"] == "PSI-25/BASS2_PMMA_RUN04@60.0s",
                "fire_status",
            ].iloc[0],
            "WATCH",
        )

    def test_legacy_case_id_does_not_override_formula_when_inputs_change(self) -> None:
        with TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                "test_id,flame_diameter_mm,burn_duration_sec,"
                "airflow_speed_cms,oxygen_concentration\n"
                "BASS_015,13.415,4.0,25.4,21\n",
            )
            result = hitung_simulasi_integrasi_nasa(path).iloc[0]

        baseline = hitung_simulasi_dataset_contoh().iloc[1]
        self.assertNotEqual(result["F_Newton"], baseline["F_Newton"])
        self.assertIsNone(result["F_reference_Newton"])

    def test_oxygen_fraction_and_optional_area_are_supported(self) -> None:
        with TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                "test_id,flame_diameter_mm,burn_duration_sec,"
                "airflow_speed_cms,oxygen_concentration,area_m2\n"
                "fraction,15.49,4,0,0.21,4\n",
            )
            result = hitung_simulasi_integrasi_nasa(path).iloc[0]

        self.assertEqual(result["oxygen_concentration_percent"], 21)
        self.assertEqual(result["rho_weight_relative_to_21pct"], 1)
        self.assertEqual(result["area"], 4)
        self.assertAlmostEqual(result["F_Newton"], 5.999001666666667)

    def test_invalid_csv_inputs_raise_clear_errors(self) -> None:
        with TemporaryDirectory() as directory:
            missing = self.write_csv(directory, "test_id,flame_diameter_mm\nBASS,2\n")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                hitung_simulasi_integrasi_nasa(missing)
            invalid = self.write_csv(
                directory,
                "test_id,flame_diameter_mm,burn_duration_sec,"
                "airflow_speed_cms,oxygen_concentration\n"
                "BASS,2,0,1,21\n",
            )
            with self.assertRaisesRegex(ValueError, "burn_duration_sec"):
                hitung_simulasi_integrasi_nasa(invalid)

    def test_custom_da2_scenario_changes_time_series_force(self) -> None:
        default = hitung_simulasi_dataset_contoh().iloc[1]
        alternate = hitung_simulasi_dataset_contoh(da2_scenario=480).iloc[1]

        self.assertNotEqual(default["F_Newton"], alternate["F_Newton"])
        self.assertEqual(alternate["DA2_units"], 480)

    def test_sensor_series_rejects_duplicate_timestamps_in_same_run(self) -> None:
        with TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                "INVESTIGATION_ID,TEST_RUN_CODE,TIME_SEC,PRESSURE_KPA,"
                "O2_CONC_PCT,CO2_CONC_PPM,FLOW_VEL_CMS,TC1_TEMP_C,"
                "TC2_TEMP_C,RADIOMETER_WM2,IGNITION_STATE\n"
                "P,R,0,101,21,400,2,20,20,0,0\n"
                "P,R,0,101,21,400,2,20,20,0,1\n",
            )
            with self.assertRaisesRegex(ValueError, "unique within each test run"):
                hitung_simulasi_integrasi_nasa(path)

    def test_active_ignition_with_zero_airflow_reports_critical(self) -> None:
        with TemporaryDirectory() as directory:
            path = self.write_csv(
                directory,
                "INVESTIGATION_ID,TEST_RUN_CODE,TIME_SEC,PRESSURE_KPA,"
                "O2_CONC_PCT,CO2_CONC_PPM,FLOW_VEL_CMS,TC1_TEMP_C,"
                "TC2_TEMP_C,RADIOMETER_WM2,IGNITION_STATE\n"
                "P,R,0,101,21,400,0,20,20,0,1\n"
                "P,R,10,101,21,400,0,20,20,0,1\n",
            )
            result = hitung_simulasi_integrasi_nasa(path).iloc[1]

        self.assertEqual(result["fire_status"], "CRITICAL")
        self.assertIn("Airflow Failure", result["fire_diagnostic"])


class FluidVelocityEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FluidVelocityEngine()

    def test_initial_dynamic_fixture_is_18_newtons(self) -> None:
        result = self.engine.calculate_sector3_status(120, 4, "0.5", 6, 1)
        self.assertEqual(
            result,
            {"Status": "STABLE", "F_Newton": 18.0, "F_int_Units": 18_000_000},
        )

    def test_manual_reference_fixture_is_22_million_units(self) -> None:
        result = self.engine.calculate_sector3_status(
            80, 2, "0.5", 4, 1, "manual_reference"
        )
        self.assertEqual(
            result,
            {"Status": "STABLE", "F_Newton": 22.0, "F_int_Units": 22_000_000},
        )

    def test_v2_5_formula_is_evaluated_literally(self) -> None:
        force = self.engine.calculate_sector3(80, 2, "0.5", 4, 1, "v2_5")
        self.assertEqual(force, Decimal(4))

    def test_zero_time_returns_critical_indicator(self) -> None:
        force = self.engine.calculate_sector3(80, 0, "0.5", 4)
        self.assertEqual(force, CRITICAL_SINGULARITY)
        self.assertEqual(
            self.engine.to_status(force),
            {"Status": "CRITICAL", "F_Newton": None, "F_int_Units": None},
        )

    def test_decimal_inputs_are_supported(self) -> None:
        force = self.engine.calculate_sector3(
            Decimal("120"), Decimal("4"), Decimal("0.5"), Decimal("6")
        )
        self.assertEqual(force, Decimal("18.00"))

    def test_non_finite_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.calculate_sector3("NaN", 1, "0.5", 4)

    def test_sector1_residual_is_zero_for_balanced_terms(self) -> None:
        residual = self.engine.sector1_momentum_residual(
            rho=1,
            d_u_dt=2,
            convective_acceleration=1,
            grad_pressure=-3,
            d_sigma_d_t=0,
            velocity=0,
            delta_temperature=0,
            length=1,
            viscosity=1,
            thermal_diffusivity=1,
            cd="0.5",
            area=4,
        )
        self.assertEqual(residual, Decimal(0))

    def test_sector2_uses_explicit_finite_k_boundary(self) -> None:
        scale_matrix = self.engine.sector2_scale_matrix(
            da2=1,
            t_flow=1,
            t_chem=1,
            phi=1,
            energy=1,
            omega_max=1,
            rho=1,
            divergence_rho_u=0,
            k_boundary=1,
        )
        self.assertEqual(scale_matrix, Decimal(1000))

    def test_earth_orbit_propagation_is_stable(self) -> None:
        times, positions = simulate_earth_orbit(samples=160)
        radii = (positions**2).sum(axis=1) ** 0.5
        self.assertEqual(positions.shape, (160, 3))
        self.assertEqual(times.shape, (160,))
        self.assertAlmostEqual(radii[0], EARTH_RADIUS_KM + 400, places=6)
        self.assertLess(float(radii.max() - radii.min()), 0.01)

    def test_orbit_rejects_out_of_range_altitude(self) -> None:
        with self.assertRaises(ValueError):
            simulate_earth_orbit(altitude_km=50)

    def test_orbit_rejects_sub_circular_speed_that_can_hit_earth(self) -> None:
        with self.assertRaises(ValueError):
            simulate_earth_orbit(speed_ratio=0.99)

    def test_orbit_circular_speed_ratio_runs(self) -> None:
        times, positions = simulate_earth_orbit(speed_ratio=1.05, samples=120)
        radii = (positions**2).sum(axis=1) ** 0.5
        self.assertEqual(len(times), 120)
        self.assertGreater(float(radii.max()), float(radii[0]))
        self.assertGreaterEqual(float(radii.min()), EARTH_RADIUS_KM + 400)


class FireSafetyTests(unittest.TestCase):
    def test_bass_example_uses_formula_result_not_project_reference(self) -> None:
        result = assess_fire_safety(
            {
                "flame_diameter_mm": 26.83,
                "burn_duration_sec": 4,
                "airflow_speed_cms": 25.4,
                "oxygen_concentration": 21,
            }
        )

        self.assertEqual(result.status, "STABLE")
        self.assertAlmostEqual(result.force_newton, 14.997481666666667)
        self.assertEqual(result.force_int_units, 14_997_482)
        self.assertEqual(result.diagnostics[0], "Parameter valid pada aturan diagnostik simulasi.")

    def test_airflow_drop_returns_critical_diagnostic_without_force(self) -> None:
        result = assess_fire_safety(
            {
                "flame_diameter_mm": 26.83,
                "burn_duration_sec": 4,
                "airflow_speed_cms": 0,
                "oxygen_concentration": 21,
            }
        )

        self.assertEqual(result.status, "CRITICAL")
        self.assertIn("Error: Ventilasi macet / Airflow Failure", result.diagnostics)
        self.assertIsNone(result.force_newton)
        self.assertIsNone(result.force_int_units)

    def test_invalid_sensor_value_is_reported_without_calculation(self) -> None:
        result = assess_fire_safety(
            {
                "flame_diameter_mm": float("nan"),
                "burn_duration_sec": 4,
                "airflow_speed_cms": 25.4,
                "oxygen_concentration": 21,
            }
        )

        self.assertEqual(result.status, "CRITICAL")
        self.assertIn("Error: Data sensor tidak finite (flame_diameter_mm)", result.diagnostics)
        self.assertIsNone(result.force_newton)

    def test_stress_fixture_exercises_multiple_alarm_checks(self) -> None:
        result = assess_fire_safety(stress_test_measurements())

        self.assertEqual(result.status, "CRITICAL")
        self.assertIn("Error: Ventilasi macet / Airflow Failure", result.diagnostics)
        self.assertGreaterEqual(len(result.diagnostics), 4)


class EarthRenderingTests(unittest.TestCase):
    def test_texture_projection_is_bounded_and_transparent_outside_globe(self) -> None:
        texture = load_earth_texture((128, 64))
        image = render_earth_orthographic(texture, size=80)

        self.assertEqual(texture.shape, (64, 128, 3))
        self.assertEqual(image.shape, (80, 80, 4))
        self.assertEqual(float(image[0, 0, 3]), 0.0)
        self.assertEqual(float(image[40, 40, 3]), 1.0)
        self.assertTrue((image >= 0).all())
        self.assertTrue((image <= 1).all())

    def test_sphere_mesh_and_cloud_alpha_have_expected_shapes(self) -> None:
        texture = load_earth_texture((128, 64))
        x, y, z, surface, clouds = earth_sphere_mesh(
            texture,
            radius_km=6371,
            longitude_samples=17,
            latitude_samples=9,
        )

        self.assertEqual(x.shape, (9, 17))
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(z.shape, x.shape)
        self.assertEqual(surface.shape, (9, 17, 4))
        self.assertEqual(clouds.shape, surface.shape)
        self.assertGreaterEqual(float(clouds[..., 3].min()), 0)
        self.assertLessEqual(float(clouds[..., 3].max()), 0.24)

    def test_orbit_visibility_classifies_front_and_back_segments(self) -> None:
        positions = np.asarray(
            ((1, 0, 0), (1, 1, 0), (-2, -1, 0)), dtype=np.float64
        )

        visible = visibility_mask(positions, np.asarray((1, 0, 0)))

        self.assertEqual(visible.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
