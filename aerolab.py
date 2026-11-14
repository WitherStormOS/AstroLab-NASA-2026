"""Independent AeroLab terminal simulator for the custom fluid formula."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from math import isfinite
from typing import Literal

K_SCALE = 1_000_000
CRITICAL_SINGULARITY = "CRITICAL: Singularity Detected"
Sector3Mode = Literal["dynamic", "v2_5", "manual_reference"]


class FluidVelocityEngine:
    """AeroLab's standalone formula engine; does not import Astrolab."""

    @staticmethod
    def _decimal(value: Decimal | int | float | str, name: str) -> Decimal:
        try:
            result = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"{name} must be a finite decimal value") from error
        if not result.is_finite() or abs(result) > Decimal("1e100"):
            raise ValueError(f"{name} must be finite and within the supported range")
        return result

    def calculate_sector3(
        self,
        da2: Decimal | int | float | str,
        delta_t: Decimal | int | float | str,
        cd: Decimal | int | float | str,
        area: Decimal | int | float | str,
        divergence: Decimal | int | float | str = 1,
        mode: Sector3Mode = "dynamic",
    ) -> Decimal | str:
        values = (
            self._decimal(da2, "DA2"),
            self._decimal(delta_t, "delta_t"),
            self._decimal(cd, "Cd"),
            self._decimal(area, "area"),
            self._decimal(divergence, "divergence"),
        )
        da2_value, dt_value, cd_value, area_value, divergence_value = values
        if mode not in ("dynamic", "v2_5", "manual_reference"):
            raise ValueError(f"Unknown Sector 3 mode: {mode}")
        if dt_value == 0:
            return CRITICAL_SINGULARITY
        if dt_value < 0:
            raise ValueError("delta_t must be positive")

        with localcontext() as context:
            context.prec = 34
            drag = cd_value * area_value
            if mode == "dynamic":
                return Decimal("0.5") * divergence_value * (da2_value / dt_value) + drag
            if mode == "v2_5":
                return (
                    Decimal("0.5")
                    * divergence_value
                    * (da2_value / Decimal(40))
                    * dt_value
                    + drag
                )
            return Decimal(20) + drag

    @staticmethod
    def to_status(force: Decimal | str) -> dict[str, str | float | int | None]:
        if force == CRITICAL_SINGULARITY:
            return {"Status": "CRITICAL", "F_Newton": None, "F_int_Units": None}
        if not isinstance(force, Decimal) or not force.is_finite():
            raise ValueError("force must be a finite Decimal or the singularity indicator")
        with localcontext() as context:
            context.prec = 34
            scaled = (force * K_SCALE).to_integral_value(rounding=ROUND_HALF_EVEN)
        converted = float(force)
        if not isfinite(converted) or (force != 0 and converted == 0):
            raise OverflowError("force cannot be represented as a finite float")
        return {
            "Status": "STABLE",
            "F_Newton": converted,
            "F_int_Units": int(scaled),
        }

    def calculate_sector3_status(
        self,
        da2: Decimal | int | float | str,
        delta_t: Decimal | int | float | str,
        cd: Decimal | int | float | str,
        area: Decimal | int | float | str,
        divergence: Decimal | int | float | str = 1,
        mode: Sector3Mode = "dynamic",
    ) -> dict[str, str | float | int | None]:
        return self.to_status(
            self.calculate_sector3(da2, delta_t, cd, area, divergence, mode)
        )


def run_cli() -> None:
    engine = FluidVelocityEngine()
    options: dict[str, Sector3Mode] = {
        "1": "dynamic",
        "2": "v2_5",
        "3": "manual_reference",
    }
    print("AeroLab | Rumus Kecepatan Fluida")
    print("Aplikasi mandiri; Astrolab dijalankan terpisah.")
    while True:
        try:
            choice = input("Formula: 1) Dinamis  2) v2.5  3) Referensi manual  (q keluar): ")
            if choice.lower() == "q":
                return
            if choice not in options:
                print("Pilih 1, 2, 3, atau q.")
                continue
            defaults = ("120", "4", "0.5", "6", "1")
            labels = ("DA2", "delta_t", "Cd", "area", "divergence")
            inputs = []
            for label, default in zip(labels, defaults):
                value = input(f"{label} [{default}]: ").strip() or default
                inputs.append(engine._decimal(value, label))
        except (EOFError, KeyboardInterrupt):
            print("\nAeroLab selesai.")
            return
        except ValueError as error:
            print(f"Input tidak valid: {error}")
            continue

        result = engine.calculate_sector3_status(*inputs, options[choice])
        if result["Status"] == "CRITICAL":
            print(CRITICAL_SINGULARITY)
        else:
            print(
                f'STATUS: STABLE | {result["F_Newton"]} N | '
                f'{result["F_int_Units"]:,} unit'
            )


if __name__ == "__main__":
    run_cli()
