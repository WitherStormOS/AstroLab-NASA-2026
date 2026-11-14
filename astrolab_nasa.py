"""Load user-supplied flame-test sensor records and calculate EM3 scenarios."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from astrolab import FluidVelocityEngine

SECTOR3_TIME_SCALE = 40
DEFAULT_AREA = 5
OXYGEN_REFERENCE_PERCENT = 21.0
AIRFLOW_CD_SPAN_CMS = 254.0
NASA_TIMESERIES_COLUMNS = (
    "INVESTIGATION_ID",
    "TEST_RUN_CODE",
    "TIME_SEC",
    "PRESSURE_KPA",
    "O2_CONC_PCT",
    "CO2_CONC_PPM",
    "FLOW_VEL_CMS",
    "TC1_TEMP_C",
    "TC2_TEMP_C",
    "RADIOMETER_WM2",
    "IGNITION_STATE",
)

REQUIRED_COLUMNS = (
    "test_id",
    "flame_diameter_mm",
    "burn_duration_sec",
    "airflow_speed_cms",
    "oxygen_concentration",
)

def hitung_simulasi_integrasi_nasa(
    file_csv_path: str | Path,
    *,
    da2_scenario: float = 240.0,
    area_m2: float = DEFAULT_AREA,
) -> pd.DataFrame:
    """Load supported NASA-style or provided sensor-series CSV data.

    Time-series rows do not contain flame diameter/DA². For those rows,
    ``da2_scenario`` is an explicit user-selected scenario parameter rather
    than a value inferred from the sensor data.
    """
    source_path = Path(file_csv_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {source_path}")

    raw = pd.read_csv(source_path)
    if set(NASA_TIMESERIES_COLUMNS).issubset(raw.columns):
        return _calculate_nasa_timeseries(
            raw,
            provenance="User-provided sensor series; source not independently verified",
            da2_scenario=da2_scenario,
            area_m2=area_m2,
        )
    return _calculate_nasa_frame(
        raw,
        provenance="User-provided legacy CSV; source not independently verified",
    )


def hitung_simulasi_dataset_contoh(
    *, da2_scenario: float = 240.0, area_m2: float = DEFAULT_AREA
) -> pd.DataFrame:
    """Calculate the user-provided flame-freefall dataset bundled with Astrolab."""
    data_path = Path(__file__).resolve().parent / "data" / "flame_freefall_timeseries.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Bundled sensor-series CSV not found: {data_path}")
    return _calculate_nasa_timeseries(
        pd.read_csv(data_path),
        provenance="User-provided flame-freefall series; not independently verified",
        da2_scenario=da2_scenario,
        area_m2=area_m2,
    )


def _calculate_nasa_timeseries(
    raw: pd.DataFrame,
    *,
    provenance: str,
    da2_scenario: float | None,
    area_m2: float,
) -> pd.DataFrame:
    """Validate time-series records and add scenario-force and airflow status."""
    missing_columns = [
        name for name in NASA_TIMESERIES_COLUMNS if name not in raw.columns
    ]
    if missing_columns:
        raise ValueError(
            "Sensor-series CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
    if raw.empty:
        raise ValueError("Sensor-series CSV contains no records")

    numeric_columns = NASA_TIMESERIES_COLUMNS[2:]
    numeric = raw.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        invalid = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(
            "Sensor-series numeric columns contain missing or invalid values: "
            + ", ".join(invalid)
        )
    if not all(
        math.isfinite(float(value))
        for row in numeric.itertuples(index=False, name=None)
        for value in row
    ):
        raise ValueError("Sensor-series numeric inputs must be finite")
    if raw.loc[:, ("INVESTIGATION_ID", "TEST_RUN_CODE")].isna().any().any():
        raise ValueError("INVESTIGATION_ID and TEST_RUN_CODE must not be empty")

    validated = raw.loc[:, NASA_TIMESERIES_COLUMNS].copy()
    validated.loc[:, numeric_columns] = numeric
    if (validated["TIME_SEC"] < 0).any():
        raise ValueError("TIME_SEC must be non-negative")
    if (validated["PRESSURE_KPA"] <= 0).any():
        raise ValueError("PRESSURE_KPA must be greater than zero")
    if ((validated["O2_CONC_PCT"] <= 0) | (validated["O2_CONC_PCT"] > 100)).any():
        raise ValueError("O2_CONC_PCT must be in (0, 100]")
    if (validated["CO2_CONC_PPM"] < 0).any():
        raise ValueError("CO2_CONC_PPM must be non-negative")
    if (validated["FLOW_VEL_CMS"] < 0).any():
        raise ValueError("FLOW_VEL_CMS must be non-negative")
    if (validated["RADIOMETER_WM2"] < 0).any():
        raise ValueError("RADIOMETER_WM2 must be non-negative")
    if not validated["IGNITION_STATE"].isin((0, 1)).all():
        raise ValueError("IGNITION_STATE must contain only 0 or 1")
    for column in ("INVESTIGATION_ID", "TEST_RUN_CODE"):
        validated[column] = validated[column].astype(str).str.strip()
        if validated[column].eq("").any():
            raise ValueError(f"{column} must not be empty")

    validated = validated.sort_values(
        ["INVESTIGATION_ID", "TEST_RUN_CODE", "TIME_SEC"], kind="stable"
    ).reset_index(drop=True)
    group_columns = ("INVESTIGATION_ID", "TEST_RUN_CODE")
    duplicate_times = validated.duplicated((*group_columns, "TIME_SEC"))
    if duplicate_times.any():
        raise ValueError("TIME_SEC must be unique within each test run")
    delta_t = validated.groupby(list(group_columns), sort=False)["TIME_SEC"].diff()

    if da2_scenario is not None:
        if not math.isfinite(float(da2_scenario)) or float(da2_scenario) < 0:
            raise ValueError("da2_scenario must be a finite non-negative number")
        da2_scenario = float(da2_scenario)
    if not math.isfinite(float(area_m2)) or float(area_m2) < 0:
        raise ValueError("area_m2 must be a finite non-negative number")
    area_m2 = float(area_m2)

    engine = FluidVelocityEngine()
    result_rows: list[dict[str, object]] = []
    for index, record in validated.iterrows():
        airflow = float(record["FLOW_VEL_CMS"])
        oxygen = float(record["O2_CONC_PCT"])
        interval = delta_t.iloc[index]
        cd = min(0.6, 0.5 + airflow / AIRFLOW_CD_SPAN_CMS)
        rho_weight = oxygen / OXYGEN_REFERENCE_PERCENT

        if interval is None or pd.isna(interval):
            force_newton = None
            force_units = None
            formula_status = "NO_PREVIOUS_SAMPLE"
        elif da2_scenario is None:
            force_newton = None
            force_units = None
            formula_status = "DA2_SCENARIO_REQUIRED"
        else:
            formula_result = engine.calculate_sector3_status(
                da2=da2_scenario,
                delta_t=float(interval),
                cd=cd,
                area=area_m2,
                divergence=1,
                mode="v2_5",
            )
            force_newton = formula_result["F_Newton"]
            force_units = formula_result["F_int_Units"]
            formula_status = str(formula_result["Status"])

        ignition = int(record["IGNITION_STATE"])
        if ignition == 0:
            fire_status = "IDLE"
            fire_diagnostic = "Ignition off; no active-flame airflow alarm."
        elif airflow == 0:
            fire_status = "CRITICAL"
            fire_diagnostic = "Error: Ventilasi macet / Airflow Failure"
        elif airflow < 5:
            fire_status = "WATCH"
            fire_diagnostic = (
                "Warning: airflow below the configurable 5 cm/s demo threshold."
            )
        else:
            fire_status = "STABLE"
            fire_diagnostic = "Ignition on; airflow passes the demo threshold."

        investigation = str(record["INVESTIGATION_ID"])
        test_run = str(record["TEST_RUN_CODE"])
        time_sec = float(record["TIME_SEC"])
        result_rows.append(
            {
                "test_id": f"{investigation}/{test_run}@{time_sec!r}s",
                "INVESTIGATION_ID": investigation,
                "TEST_RUN_CODE": test_run,
                "TIME_SEC": time_sec,
                "PRESSURE_KPA": float(record["PRESSURE_KPA"]),
                "O2_CONC_PCT": oxygen,
                "CO2_CONC_PPM": float(record["CO2_CONC_PPM"]),
                "FLOW_VEL_CMS": airflow,
                "TC1_TEMP_C": float(record["TC1_TEMP_C"]),
                "TC2_TEMP_C": float(record["TC2_TEMP_C"]),
                "RADIOMETER_WM2": float(record["RADIOMETER_WM2"]),
                "IGNITION_STATE": ignition,
                "DA2_units": da2_scenario,
                "DA2_is_scenario_assumption": True,
                "burn_duration_sec": (
                    float(interval) if interval is not None and not pd.isna(interval) else None
                ),
                "airflow_speed_cms": airflow,
                "Cd": cd,
                "area": area_m2,
                "oxygen_concentration_percent": oxygen,
                "rho_weight_relative_to_21pct": rho_weight,
                "F_formula_Newton": force_newton,
                "F_formula_int_Units": force_units,
                "F_reference_Newton": None,
                "F_reference_int_Units": None,
                "F_Newton": force_newton,
                "F_int_Units": force_units,
                "Status": formula_status,
                "fire_status": fire_status,
                "fire_diagnostic": fire_diagnostic,
                "output_kind": "time-series scenario",
                "data_kind": "timeseries",
                "data_provenance": provenance,
            }
        )
    return pd.DataFrame(result_rows)


def _calculate_nasa_frame(raw: pd.DataFrame, provenance: str) -> pd.DataFrame:
    """Validate NASA-style rows and calculate formula and reference outputs."""
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in raw.columns]
    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing_columns)
        )
    if raw.empty:
        raise ValueError("CSV contains no simulation records")

    numeric_columns = (
        "flame_diameter_mm",
        "burn_duration_sec",
        "airflow_speed_cms",
        "oxygen_concentration",
    )
    numeric = raw.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    values = [
        tuple(float(value) for value in row)
        for row in numeric.itertuples(index=False, name=None)
    ]
    if any(not math.isfinite(value) for row in values for value in row):
        raise ValueError("CSV numeric inputs must contain finite numbers")

    diameter = [row[0] for row in values]
    duration = [row[1] for row in values]
    airflow = [row[2] for row in values]
    oxygen_input = [row[3] for row in values]
    if any(value < 0 for value in diameter):
        raise ValueError("flame_diameter_mm must be non-negative")
    if any(value <= 0 for value in duration):
        raise ValueError("burn_duration_sec must be greater than zero")
    if any(value < 0 for value in airflow):
        raise ValueError("airflow_speed_cms must be non-negative")

    oxygen_percent = [value * 100 if value <= 1 else value for value in oxygen_input]
    if any(value <= 0 or value > 100 for value in oxygen_percent):
        raise ValueError(
            "oxygen_concentration must be in (0, 1] fraction or (0, 100] percent"
        )

    # NASA bridge conversion: diameter^2 is mapped to the DA² display variable.
    da2_units = [value * value / 3.0 for value in diameter]
    cd_values = [
        min(max(0.5 + value / AIRFLOW_CD_SPAN_CMS, 0.5), 0.6)
        for value in airflow
    ]
    rho_weights = [value / OXYGEN_REFERENCE_PERCENT for value in oxygen_percent]

    area_column = next(
        (name for name in ("area_m2", "area") if name in raw.columns), None
    )
    if area_column is None:
        area_values = [float(DEFAULT_AREA)] * len(raw)
    else:
        area_values = [
            float(value)
            for value in pd.to_numeric(raw[area_column], errors="coerce").tolist()
        ]
        if any(not math.isfinite(value) or value < 0 for value in area_values):
            raise ValueError(f"{area_column} must contain finite non-negative numbers")

    engine = FluidVelocityEngine()
    result_rows: list[dict[str, object]] = []
    for index, (_, record) in enumerate(raw.iterrows()):
        test_id = str(record["test_id"])
        da2_value = str(da2_units[index])
        delta_t_value = str(duration[index])
        cd_value = str(cd_values[index])
        area_value = str(area_values[index])

        status = engine.calculate_sector3_status(
            da2=da2_value,
            delta_t=delta_t_value,
            cd=cd_value,
            area=area_value,
            divergence=1,
            mode="v2_5",
        )

        formula_force_newton = (
            float(status["F_Newton"]) if status["F_Newton"] is not None else None
        )
        formula_force_int_units = (
            int(status["F_int_Units"]) if status["F_int_Units"] is not None else None
        )
        result_rows.append(
            {
                "test_id": test_id,
                "flame_diameter_mm": diameter[index],
                "DA2_units": da2_units[index],
                "burn_duration_sec": duration[index],
                "airflow_speed_cms": airflow[index],
                "Cd": cd_values[index],
                "area": area_values[index],
                "oxygen_concentration_percent": oxygen_percent[index],
                "rho_weight_relative_to_21pct": rho_weights[index],
                "F_formula_Newton": formula_force_newton,
                "F_formula_int_Units": formula_force_int_units,
                "F_reference_Newton": None,
                "F_reference_int_Units": None,
                "F_Newton": formula_force_newton,
                "F_int_Units": formula_force_int_units,
                "Status": status["Status"],
                "output_kind": "formula",
                "data_kind": "legacy",
                "data_provenance": provenance,
            }
        )
    return pd.DataFrame(result_rows)
