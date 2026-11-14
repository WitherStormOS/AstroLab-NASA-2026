"""Fast input-health diagnostics for the educational fire scenario panel."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from time import perf_counter_ns
from typing import Mapping

from astrolab import FluidVelocityEngine

DEFAULT_AREA_M2 = Decimal("5")
AIRFLOW_CD_SPAN_CMS = Decimal("254")
MAX_FLAME_DIAMETER_MM = 1000.0
MAX_BURN_DURATION_SEC = 3600.0
MAX_AIRFLOW_SPEED_CMS = 1000.0
MIN_DEMO_AIRFLOW_SPEED_CMS = 5.0


@dataclass(frozen=True)
class FireSafetyAssessment:
    status: str
    diagnostics: tuple[str, ...]
    oxygen_percent: float | None
    force_newton: float | None
    force_int_units: int | None
    latency_ms: float


def assess_fire_safety(
    measurements: Mapping[str, object],
) -> FireSafetyAssessment:
    """Check sensor-input sanity and calculate the custom force when valid.

    Limits in this function are defensive UI-simulation bounds, not NASA flight
    limits or a validated spacecraft fire-detection model.
    """
    started_ns = perf_counter_ns()
    diagnostics: list[str] = []
    required = (
        "flame_diameter_mm",
        "burn_duration_sec",
        "airflow_speed_cms",
        "oxygen_concentration",
    )
    values: dict[str, float] = {}

    for name in required:
        raw_value = measurements.get(name)
        try:
            if isinstance(raw_value, bool):
                raise ValueError
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            diagnostics.append(f"Error: Data sensor tidak valid ({name})")
            continue
        if not isfinite(numeric_value):
            diagnostics.append(f"Error: Data sensor tidak finite ({name})")
            continue
        values[name] = numeric_value

    if len(values) != len(required):
        return FireSafetyAssessment(
            status="CRITICAL",
            diagnostics=tuple(diagnostics),
            oxygen_percent=None,
            force_newton=None,
            force_int_units=None,
            latency_ms=(perf_counter_ns() - started_ns) / 1_000_000,
        )

    diameter = values["flame_diameter_mm"]
    duration = values["burn_duration_sec"]
    airflow = values["airflow_speed_cms"]
    oxygen_percent = values["oxygen_concentration"]
    if 0 < oxygen_percent <= 1:
        oxygen_percent *= 100

    if not 0 < diameter <= MAX_FLAME_DIAMETER_MM:
        diagnostics.append(
            "Error: Diameter api di luar batas input simulasi (0–1.000 mm)"
        )
    if not 0 < duration <= MAX_BURN_DURATION_SEC:
        diagnostics.append(
            "Error: Durasi pembakaran di luar batas input simulasi (0–3.600 s)"
        )
    if airflow < MIN_DEMO_AIRFLOW_SPEED_CMS:
        diagnostics.append("Error: Ventilasi macet / Airflow Failure")
    elif airflow > MAX_AIRFLOW_SPEED_CMS:
        diagnostics.append(
            "Error: Laju aliran di luar batas input simulasi (maks. 1.000 cm/s)"
        )
    if not 0 < oxygen_percent <= 100:
        diagnostics.append(
            "Error: Konsentrasi O₂ di luar rentang input simulasi (0–100%)"
        )

    force_newton = None
    force_int_units = None
    if not diagnostics:
        diameter_decimal = Decimal(str(diameter))
        da2 = diameter_decimal * diameter_decimal / Decimal(3)
        airflow_decimal = Decimal(str(airflow))
        cd = min(
            Decimal("0.6"),
            Decimal("0.5") + airflow_decimal / AIRFLOW_CD_SPAN_CMS,
        )
        result = FluidVelocityEngine().calculate_sector3_status(
            da2=da2,
            delta_t=Decimal(str(duration)),
            cd=cd,
            area=DEFAULT_AREA_M2,
            divergence=1,
            mode="v2_5",
        )
        force_newton = float(result["F_Newton"])
        force_int_units = int(result["F_int_Units"])
        diagnostics.append("Parameter valid pada aturan diagnostik simulasi.")

    return FireSafetyAssessment(
        status="STABLE" if force_newton is not None else "CRITICAL",
        diagnostics=tuple(diagnostics),
        oxygen_percent=oxygen_percent,
        force_newton=force_newton,
        force_int_units=force_int_units,
        latency_ms=(perf_counter_ns() - started_ns) / 1_000_000,
    )


def stress_test_measurements() -> dict[str, float]:
    """Return a synthetic multi-fault profile for exercising alarm handling."""
    return {
        "flame_diameter_mm": 1_000_000_000.0,
        "burn_duration_sec": 0.0,
        "airflow_speed_cms": 0.0,
        "oxygen_concentration": 250.0,
    }
