"""Astrolab: lightweight desktop app for user-defined fluid equations.

The custom equations and example profiles in this module are not validated
aerospace models and must not be used for vehicle design or flight control.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from math import exp, isfinite
from pathlib import Path
from typing import Literal

from astrolab_simulations import (
    EARTH_MU_KM3_S2,
    EARTH_RADIUS_KM,
    simulate_earth_orbit,
)

K_SCALE = 1_000_000
MAX_INPUT_MAGNITUDE = Decimal("1e100")
PI = Decimal("3.141592653589793238462643383279503")
CRITICAL_SINGULARITY = "CRITICAL: Singularity Detected"
Sector3Mode = Literal["dynamic", "v2_5", "manual_reference"]


class FluidVelocityEngine:
    """Evaluate the custom scalar formulas with Decimal arithmetic."""

    def __init__(self, k_scale: int = K_SCALE) -> None:
        if (
            not isinstance(k_scale, int)
            or isinstance(k_scale, bool)
            or not 0 < k_scale <= 10**12
        ):
            raise ValueError("k_scale must be an integer from 1 to 1,000,000,000,000")
        self.k_scale = k_scale

    @staticmethod
    def _decimal(value: Decimal | int | float | str, name: str) -> Decimal:
        try:
            result = value if isinstance(value, Decimal) else Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"{name} must be a finite decimal value") from error
        if not result.is_finite():
            raise ValueError(f"{name} must be a finite decimal value")
        if abs(result) > MAX_INPUT_MAGNITUDE:
            raise ValueError(f"{name} exceeds the supported magnitude")
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
        """Return the scalar force in N, or the required singularity indicator.

        ``dynamic`` implements the original formula. ``v2_5`` implements the
        later equation literally. ``manual_reference`` preserves the separate
        Chief Engineer reference calculation with its fixed 0.5 * 40 base.
        """
        da2_value = self._decimal(da2, "da2")
        delta_t_value = self._decimal(delta_t, "delta_t")
        cd_value = self._decimal(cd, "cd")
        area_value = self._decimal(area, "area")
        divergence_value = self._decimal(divergence, "divergence")

        if mode not in ("dynamic", "v2_5", "manual_reference"):
            raise ValueError(f"Unknown Sector 3 mode: {mode}")
        if delta_t_value == 0:
            return CRITICAL_SINGULARITY
        if delta_t_value < 0:
            raise ValueError("delta_t must be positive")

        with localcontext() as context:
            context.prec = 34
            drag = cd_value * area_value
            if mode == "dynamic":
                # F = 0.5 * divergence * (DA² / delta_t) + Cd * A
                return (
                    Decimal("0.5")
                    * divergence_value
                    * (da2_value / delta_t_value)
                    + drag
                )
            if mode == "v2_5":
                # Literal form: F = 0.5 * grad * (DA² / 40) * delta_t + Cd * A.
                return (
                    Decimal("0.5")
                    * divergence_value
                    * (da2_value / Decimal(40))
                    * delta_t_value
                    + drag
                )

            # The supplied 22 N hand-check uses 0.5 * 40 as its base;
            # that base is not algebraically equivalent to either equation above.
            return Decimal("0.5") * Decimal(40) + drag

    def to_status(self, force: Decimal | str) -> dict[str, str | float | int | None]:
        """Convert Newtons to the public status dictionary and scaled units."""
        if force == CRITICAL_SINGULARITY:
            return {"Status": "CRITICAL", "F_Newton": None, "F_int_Units": None}
        if not isinstance(force, Decimal) or not force.is_finite():
            raise ValueError("force must be a finite Decimal or the singularity indicator")

        with localcontext() as context:
            context.prec = 34
            scaled_force = (force * self.k_scale).to_integral_value(
                rounding=ROUND_HALF_EVEN
            )
        newton_force = float(force)
        if not isfinite(newton_force) or (force != 0 and newton_force == 0):
            raise OverflowError("force cannot be represented as a finite float")
        return {
            "Status": "STABLE",
            "F_Newton": newton_force,
            "F_int_Units": int(scaled_force),
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
        force = self.calculate_sector3(da2, delta_t, cd, area, divergence, mode)
        return self.to_status(force)

    def sector1_momentum_residual(
        self,
        rho: Decimal | int | float | str,
        d_u_dt: Decimal | int | float | str,
        convective_acceleration: Decimal | int | float | str,
        grad_pressure: Decimal | int | float | str,
        d_sigma_d_t: Decimal | int | float | str,
        velocity: Decimal | int | float | str,
        delta_temperature: Decimal | int | float | str,
        length: Decimal | int | float | str,
        viscosity: Decimal | int | float | str,
        thermal_diffusivity: Decimal | int | float | str,
        cd: Decimal | int | float | str,
        area: Decimal | int | float | str,
    ) -> Decimal:
        """Return inertial term minus RHS for the scalar Sector 1 equation."""
        values = {
            name: self._decimal(value, name)
            for name, value in {
                "rho": rho,
                "d_u_dt": d_u_dt,
                "convective_acceleration": convective_acceleration,
                "grad_pressure": grad_pressure,
                "d_sigma_d_t": d_sigma_d_t,
                "velocity": velocity,
                "delta_temperature": delta_temperature,
                "length": length,
                "viscosity": viscosity,
                "thermal_diffusivity": thermal_diffusivity,
                "cd": cd,
                "area": area,
            }.items()
        }
        if values["rho"] <= 0:
            raise ValueError("rho must be positive")
        if values["viscosity"] == 0 or values["thermal_diffusivity"] == 0:
            raise ZeroDivisionError("viscosity and thermal_diffusivity must be non-zero")
        if values["viscosity"] < 0 or values["thermal_diffusivity"] < 0:
            raise ValueError("viscosity and thermal_diffusivity must be positive")

        with localcontext() as context:
            context.prec = 34
            inertial = values["rho"] * (
                values["d_u_dt"] + values["convective_acceleration"]
            )
            rhs = (
                -values["grad_pressure"]
                + values["d_sigma_d_t"]
                * Decimal("0.5")
                * values["rho"]
                * values["velocity"]
                * values["velocity"]
                + (
                    values["delta_temperature"]
                    * values["length"]
                    / (values["viscosity"] * values["thermal_diffusivity"])
                    * values["cd"]
                    * values["area"]
                )
            )
            return inertial - rhs

    def sector2_scale_matrix(
        self,
        da2: Decimal | int | float | str,
        t_flow: Decimal | int | float | str,
        t_chem: Decimal | int | float | str,
        phi: Decimal | int | float | str,
        energy: Decimal | int | float | str,
        omega_max: Decimal | int | float | str,
        rho: Decimal | int | float | str,
        divergence_rho_u: Decimal | int | float | str,
        k_boundary: Decimal | int | float | str,
    ) -> Decimal:
        """Evaluate Sector 2 with an explicit finite approximation for k-limit.

        ``k_boundary`` is a user-selected finite substitute for the undefined
        literal ``lim k -> infinity``; it is not the mathematical limit.
        """
        values = {
            name: self._decimal(value, name)
            for name, value in {
                "da2": da2,
                "t_flow": t_flow,
                "t_chem": t_chem,
                "phi": phi,
                "energy": energy,
                "omega_max": omega_max,
                "rho": rho,
                "divergence_rho_u": divergence_rho_u,
                "k_boundary": k_boundary,
            }.items()
        }
        if values["t_chem"] == 0 or values["rho"] == 0:
            raise ZeroDivisionError("t_chem and rho must be non-zero")
        if values["t_chem"] < 0 or values["rho"] < 0:
            raise ValueError("t_chem and rho must be positive")
        if values["t_flow"] < 0 or values["omega_max"] <= 0:
            raise ValueError("t_flow must be non-negative and omega_max positive")
        if values["energy"] <= 0:
            raise ValueError("energy must be positive for the real-valued power")
        if values["k_boundary"] < 0:
            raise ValueError("k_boundary must be non-negative")

        with localcontext() as context:
            context.prec = 34
            omega_per_density = values["omega_max"] / values["rho"]
            if omega_per_density <= 0:
                raise ValueError(
                    "omega_max / rho must be positive for the real square root"
                )
            energy_power = values["energy"] ** PI
            radicand = energy_power * self.k_scale / omega_per_density
            return (
                values["da2"]
                * (values["t_flow"] / values["t_chem"])
                * values["phi"]
                * radicand.sqrt()
                * (values["k_boundary"] + values["divergence_rho_u"])
            )


def _run_static_verification() -> None:
    engine = FluidVelocityEngine()
    initial = engine.calculate_sector3_status(120, 4, "0.5", 6, 1, "dynamic")
    manual = engine.calculate_sector3_status(80, 2, "0.5", 4, 1, "manual_reference")
    literal_v2_5 = engine.calculate_sector3_status(80, 2, "0.5", 4, 1, "v2_5")
    singular = engine.calculate_sector3_status(80, 0, "0.5", 4, 1, "dynamic")

    assert initial == {
        "Status": "STABLE",
        "F_Newton": 18.0,
        "F_int_Units": 18_000_000,
    }
    assert manual == {
        "Status": "STABLE",
        "F_Newton": 22.0,
        "F_int_Units": 22_000_000,
    }
    assert literal_v2_5["F_Newton"] == 4.0
    assert singular == {"Status": "CRITICAL", "F_Newton": None, "F_int_Units": None}


def run_cli() -> None:
    """Run the interactive terminal version without graphical dependencies."""
    engine = FluidVelocityEngine()
    print("Astrolab | Rumus Kecepatan Fluida")
    print("Simulator formula kustom; bukan model aerospace tervalidasi.")
    print("Pilih formula: 1) Dinamis  2) v2.5 literal  3) Referensi manual 22 N")

    mode_options: dict[str, Sector3Mode] = {
        "1": "dynamic",
        "2": "v2_5",
        "3": "manual_reference",
    }

    def prompt_decimal(label: str, default: str) -> Decimal:
        while True:
            raw_value = input(f"{label} [{default}]: ").strip() or default
            try:
                return engine._decimal(raw_value, label)
            except ValueError as error:
                print(f"Input tidak valid: {error}")

    while True:
        try:
            selected = input("Pilihan formula [1/2/3, q untuk keluar] (1): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAstrolab selesai.")
            return
        if selected.lower() == "q":
            print("Astrolab selesai.")
            return
        selected = selected or "1"
        if selected not in mode_options:
            print("Pilihan tidak dikenal. Masukkan 1, 2, 3, atau q.")
            continue

        mode = mode_options[selected]
        try:
            da2 = prompt_decimal("DA2", "80" if mode != "dynamic" else "120")
            delta_t = prompt_decimal("delta_t", "2" if mode != "dynamic" else "4")
            cd = prompt_decimal("Cd", "0.5")
            area = prompt_decimal("Area", "4" if mode != "dynamic" else "6")
            divergence = prompt_decimal("Divergence / grad", "1")
        except (EOFError, KeyboardInterrupt):
            print("\nAstrolab selesai.")
            return
        result = engine.calculate_sector3_status(
            da2, delta_t, cd, area, divergence, mode
        )
        print(f'Status: {result["Status"]}')
        if result["Status"] == "CRITICAL":
            print(CRITICAL_SINGULARITY)
        else:
            print(f'Gaya: {result["F_Newton"]} N')
            print(f'F_int: {result["F_int_Units"]:,} unit')


def run_gui() -> None:
    """Run Astrolab as a compact native desktop application."""
    import tkinter as tk
    from tkinter import ttk

    import numpy as np
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

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

    engine = FluidVelocityEngine()
    language = {"code": "id"}

    def tr(text: str) -> str:
        return translate(text, language["code"])

    colors = {
        "background": "#0b1220",
        "surface": "#111c2f",
        "surface_alt": "#17243a",
        "border": "#25344d",
        "text": "#edf3fb",
        "muted": "#9aabc2",
        "accent": "#47d7ac",
        "accent_dark": "#183b3a",
        "blue": "#62a8ff",
        "orange": "#ff9b71",
        "danger": "#ff7777",
        "input": "#0d1728",
    }

    root = tk.Tk()
    root.title("Astrolab | Rumus Kecepatan Fluida")
    root.geometry("1280x920")
    root.minsize(800, 640)
    root.configure(background=colors["background"])

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "Astrolab.TCombobox",
        fieldbackground=colors["input"],
        background=colors["surface_alt"],
        foreground=colors["text"],
        arrowcolor=colors["accent"],
        bordercolor=colors["border"],
        lightcolor=colors["border"],
        darkcolor=colors["border"],
        padding=8,
    )
    style.map(
        "Astrolab.TCombobox",
        fieldbackground=[("readonly", colors["input"])],
        foreground=[("readonly", colors["text"])],
        selectbackground=[("readonly", colors["input"])],
        selectforeground=[("readonly", colors["text"])],
    )
    style.configure(
        "Astrolab.TCheckbutton",
        background=colors["surface"],
        foreground=colors["text"],
        font=("Segoe UI", 10),
        padding=4,
    )
    style.map(
        "Astrolab.TCheckbutton",
        background=[("active", colors["surface"])],
        foreground=[("active", colors["accent"])],
    )

    def label(parent, text, *, size=10, color=None, weight="normal"):
        return tk.Label(
            parent,
            text=tr(text),
            bg=parent.cget("background"),
            fg=color or colors["text"],
            font=("Segoe UI", size, weight),
            anchor="w",
        )

    def card(parent, *, padx=18, pady=16):
        frame = tk.Frame(
            parent,
            bg=colors["surface"],
            highlightbackground=colors["border"],
            highlightthickness=1,
            padx=padx,
            pady=pady,
        )
        return frame

    shell = tk.Frame(root, bg=colors["background"], padx=26, pady=20)
    shell.pack(fill="both", expand=True)
    shell.columnconfigure(0, weight=1)
    shell.rowconfigure(3, weight=1)

    header = tk.Frame(shell, bg=colors["background"])
    header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
    header.columnconfigure(0, weight=1)
    brand = tk.Frame(header, bg=colors["background"])
    brand.grid(row=0, column=0, sticky="w")
    tk.Label(
        brand,
        text="A",
        bg=colors["accent_dark"],
        fg=colors["accent"],
        font=("Segoe UI", 17, "bold"),
        width=2,
        height=1,
        padx=3,
        pady=2,
    ).pack(side="left", padx=(0, 12))
    brand_text = tk.Frame(brand, bg=colors["background"])
    brand_text.pack(side="left")
    label(brand_text, "ASTROLAB", size=22, weight="bold").pack(anchor="w")
    label(
        brand_text,
        "LABORATORIUM RUMUS KECEPATAN FLUIDA",
        size=9,
        color=colors["muted"],
    ).pack(anchor="w", pady=(1, 0))
    status_badge = tk.Label(
        header,
        text="●  STABLE",
        bg=colors["accent_dark"],
        fg=colors["accent"],
        font=("Segoe UI", 9, "bold"),
        padx=14,
        pady=8,
    )
    status_badge.grid(row=0, column=1, sticky="e")

    intro = label(
        shell,
        "Simulator ringan untuk mengeksplorasi parameter dan formula kustom.",
        color=colors["muted"],
    )
    intro.grid(row=1, column=0, sticky="w", pady=(0, 16))

    formula_card = card(shell, padx=18, pady=14)
    formula_card.grid(row=2, column=0, sticky="ew", pady=(0, 14))
    formula_card.columnconfigure(1, weight=1)
    label(formula_card, "SEKTOR 3", size=9, color=colors["accent"], weight="bold").grid(
        row=0, column=0, sticky="w", pady=(0, 6)
    )
    mode_var = tk.StringVar(value=tr("Formula dinamis awal"))
    mode_select = ttk.Combobox(
        formula_card,
        textvariable=mode_var,
        values=(
            tr("Formula dinamis awal"),
            tr("Persamaan v2.5 literal"),
            tr("Referensi manual 22 N"),
        ),
        state="readonly",
        style="Astrolab.TCombobox",
        width=27,
    )
    mode_select.grid(row=1, column=0, sticky="w")
    label(
        formula_card,
        "F = 0.5 × divergensi × (DA² / delta_t) + Cd × A",
        size=10,
        color=colors["muted"],
    ).grid(row=1, column=1, sticky="w", padx=(18, 0))
    label(
        formula_card,
        "Pilih interpretasi formula; hasil dan grafik diperbarui saat parameter berubah.",
        size=9,
        color=colors["muted"],
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    main_area = tk.Frame(shell, bg=colors["background"])
    main_area.grid(row=3, column=0, sticky="nsew", pady=(0, 14))
    main_area.columnconfigure(0, weight=1)
    main_area.rowconfigure(1, weight=1)

    navigation = tk.Frame(main_area, bg=colors["background"])
    navigation.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    page_container = tk.Frame(main_area, bg=colors["background"])
    page_container.grid(row=1, column=0, sticky="nsew")
    page_container.columnconfigure(0, weight=1)
    page_container.rowconfigure(0, weight=1)

    home_page = tk.Frame(page_container, bg=colors["background"])
    nasa_page = tk.Frame(page_container, bg=colors["background"])
    fire_page = tk.Frame(page_container, bg=colors["background"])
    rocket_page = tk.Frame(page_container, bg=colors["background"])
    pages = {
        "Beranda": home_page,
        "Data NASA": nasa_page,
        "Roket vakum": rocket_page,
        "Fire Safety": fire_page,
    }
    for page in pages.values():
        page.grid(row=0, column=0, sticky="nsew")

    home_page.columnconfigure(0, weight=1)
    home_page.rowconfigure(0, weight=1)
    home_scroll_canvas = tk.Canvas(
        home_page,
        bg=colors["background"],
        highlightthickness=0,
        borderwidth=0,
    )
    home_scrollbar = ttk.Scrollbar(
        home_page, orient="vertical", command=home_scroll_canvas.yview
    )
    home_scroll_canvas.configure(yscrollcommand=home_scrollbar.set)
    home_scroll_canvas.grid(row=0, column=0, sticky="nsew")
    home_scrollbar.grid(row=0, column=1, sticky="ns")
    home_scroll_frame = tk.Frame(home_scroll_canvas, bg=colors["background"])
    home_scroll_window = home_scroll_canvas.create_window(
        (0, 0), window=home_scroll_frame, anchor="nw"
    )
    home_scroll_frame.columnconfigure(0, weight=1)
    home_scroll_frame.bind(
        "<Configure>",
        lambda _event: home_scroll_canvas.configure(
            scrollregion=home_scroll_canvas.bbox("all")
        ),
    )
    home_scroll_canvas.bind(
        "<Configure>",
        lambda event: home_scroll_canvas.itemconfigure(
            home_scroll_window, width=event.width
        ),
    )
    root.bind_all(
        "<MouseWheel>",
        lambda event: home_scroll_canvas.yview_scroll(
            int(-event.delta / 120), "units"
        ),
    )
    content = tk.Frame(home_scroll_frame, bg=colors["background"])
    content.grid(row=0, column=0, sticky="ew", pady=(0, 14))
    content.columnconfigure(0, weight=5, uniform="content")
    content.columnconfigure(1, weight=4, uniform="content")
    content.rowconfigure(0, weight=1)

    parameter_card = card(content, padx=20, pady=18)
    parameter_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    result_card = card(content, padx=20, pady=18)
    result_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    label(parameter_card, "Parameter simulasi", size=14, weight="bold").grid(
        row=0, column=0, columnspan=2, sticky="w"
    )
    label(
        parameter_card,
        "Masukkan nilai untuk menghitung gaya skalar.",
        color=colors["muted"],
        size=9,
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 14))

    default_values = {
        "DA²": "120",
        "delta_t": "4",
        "Cd": "0.5",
        "Area": "6",
        "Divergensi / grad": "1",
    }
    units = {
        "DA²": "unit",
        "delta_t": "s",
        "Cd": "—",
        "Area": "m²",
        "Divergensi / grad": "—",
    }
    parameter_vars: dict[str, tk.StringVar] = {}
    for row, name in enumerate(default_values, start=2):
        label(parameter_card, name, size=10).grid(row=row, column=0, sticky="w", pady=6)
        input_cell = tk.Frame(parameter_card, bg=colors["surface"])
        input_cell.grid(row=row, column=1, sticky="ew", pady=5)
        parameter_card.columnconfigure(1, weight=1)
        var = tk.StringVar(value=default_values[name])
        parameter_vars[name] = var
        entry = tk.Entry(
            input_cell,
            textvariable=var,
            bg=colors["input"],
            fg=colors["text"],
            insertbackground=colors["accent"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=colors["border"],
            highlightcolor=colors["accent"],
            font=("Segoe UI", 10),
            justify="right",
        )
        entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(0, 8))
        tk.Label(
            input_cell,
            text=units[name],
            bg=colors["surface"],
            fg=colors["muted"],
            font=("Segoe UI", 9),
            width=4,
            anchor="e",
        ).pack(side="right")

    label(result_card, "Hasil perhitungan", size=14, weight="bold").pack(anchor="w")
    label(
        result_card,
        "Keluaran skalar sektor gaya",
        size=9,
        color=colors["muted"],
    ).pack(anchor="w", pady=(4, 14))

    force_panel = tk.Frame(result_card, bg=colors["surface_alt"], padx=14, pady=12)
    force_panel.pack(fill="x")
    label(force_panel, "GAYA AKHIR", size=9, color=colors["muted"], weight="bold").pack(
        anchor="w"
    )
    force_value = tk.StringVar(value="—")
    tk.Label(
        force_panel,
        textvariable=force_value,
        bg=colors["surface_alt"],
        fg=colors["text"],
        font=("Segoe UI", 28, "bold"),
    ).pack(anchor="w", pady=(4, 0))

    metric_row = tk.Frame(result_card, bg=colors["background"], pady=0)
    metric_row.pack(fill="x", pady=(10, 0))

    def stat_chip(parent, label_text, value_text, accent=colors["accent"]):
        chip = tk.Frame(parent, bg=colors["surface"], padx=10, pady=8, highlightbackground=colors["border"], highlightthickness=1)
        chip.pack(side="left", fill="y", expand=True, padx=(0, 8))
        tk.Label(chip, text=label_text, bg=colors["surface"], fg=colors["muted"], font=("Segoe UI", 8, "bold")).pack(anchor="w")
        tk.Label(chip, text=value_text, bg=colors["surface"], fg=accent, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(2, 0))
        return chip

    stat_chip(metric_row, "MODE", "DINAMIS", colors["blue"])
    stat_chip(metric_row, "K-SCALE", "1.000.000", colors["accent"])
    stat_chip(metric_row, "SEKTOR", "3 / FLUIDA", colors["orange"])

    scale_panel = tk.Frame(result_card, bg=colors["surface"], pady=12)
    scale_panel.pack(fill="x", pady=(10, 0))
    label(scale_panel, "F_int  ·  K_scale = 1.000.000", size=9, color=colors["muted"]).pack(
        anchor="w"
    )
    units_value = tk.StringVar(value="—")
    tk.Label(
        scale_panel,
        textvariable=units_value,
        bg=colors["surface"],
        fg=colors["accent"],
        font=("Segoe UI", 17, "bold"),
    ).pack(anchor="w", pady=(3, 0))

    output_message = tk.StringVar(value="Nilai stabil dihitung dengan Decimal.")
    message_label = tk.Label(
        result_card,
        textvariable=output_message,
        bg=colors["surface"],
        fg=colors["muted"],
        font=("Segoe UI", 9),
        justify="left",
        wraplength=300,
        anchor="w",
    )
    message_label.pack(fill="x", pady=(10, 12))

    quick_check = tk.Frame(result_card, bg=colors["surface"], padx=12, pady=10)
    quick_check.pack(fill="x")
    label(quick_check, "Cek cepat formula", size=10, weight="bold").pack(anchor="w")
    label(quick_check, "Dinamis  18 N  ·  18.000.000 unit", size=9, color=colors["muted"]).pack(anchor="w", pady=(6, 2))
    label(quick_check, "Referensi manual  22 N  ·  22.000.000 unit", size=9, color=colors["muted"]).pack(anchor="w", pady=2)
    label(quick_check, "v2.5 literal  4 N  ·  4.000.000 unit", size=9, color=colors["muted"]).pack(anchor="w", pady=2)

    nasa_page.columnconfigure(0, weight=1)
    nasa_page.rowconfigure(0, weight=1)
    nasa_card = card(nasa_page, padx=18, pady=14)
    nasa_card.grid(row=0, column=0, sticky="nsew")
    nasa_card.rowconfigure(5, weight=1)
    nasa_card.columnconfigure(0, weight=3)
    nasa_card.columnconfigure(1, weight=2)
    nasa_da2_var = tk.StringVar(value="240")
    nasa_header = tk.Frame(nasa_card, bg=colors["surface"])
    nasa_header.grid(row=0, column=0, columnspan=2, sticky="ew")
    label(nasa_header, "Integrasi CSV eksperimen", size=13, weight="bold").pack(
        side="left"
    )
    nasa_provenance_note = label(
        nasa_card,
        "Data sensor yang Anda berikan: dua test run, sepuluh sampel. Asal resmi belum diverifikasi aplikasi ini; CSV lain dapat diimpor.",
        size=9,
        color=colors["muted"],
    )
    nasa_provenance_note.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 9))
    nasa_formula_note = label(
        nasa_card,
        "Δt berasal dari selang waktu antarrekaman. Data tidak memuat diameter api; DA² adalah asumsi skenario, bukan hasil ukur.",
        size=9,
        color=colors["muted"],
    )
    nasa_formula_note.configure(wraplength=420, justify="left")
    nasa_formula_note.grid(row=2, column=0, sticky="w", pady=(0, 8), padx=(0, 12))
    nasa_da2_frame = tk.Frame(nasa_card, bg=colors["surface"])
    nasa_da2_frame.grid(row=2, column=1, sticky="e", pady=(0, 8))
    label(nasa_da2_frame, "Asumsi DA²", size=8, color=colors["muted"]).pack(
        side="left", padx=(0, 5)
    )
    tk.Entry(
        nasa_da2_frame,
        textvariable=nasa_da2_var,
        width=9,
        bg=colors["input"],
        fg=colors["text"],
        insertbackground=colors["accent"],
        relief="flat",
        justify="right",
    ).pack(side="left", ipady=4)
    nasa_results: list[dict[str, object]] = []
    nasa_source_state: dict[str, Path | None] = {"path": None}
    nasa_case_var = tk.StringVar(value="")
    nasa_case_select = ttk.Combobox(
        nasa_card,
        textvariable=nasa_case_var,
        state="readonly",
        style="Astrolab.TCombobox",
        width=24,
    )
    nasa_case_select.grid(row=3, column=0, sticky="w", pady=(0, 8))
    nasa_upload_message = tk.StringVar(value="Memuat rangkaian sensor...")
    nasa_upload_label = tk.Label(
        nasa_card,
        textvariable=nasa_upload_message,
        bg=colors["surface"],
        fg=colors["muted"],
        font=("Segoe UI", 9),
        anchor="w",
    )
    nasa_upload_label.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 8))
    nasa_force_var = tk.StringVar(value="—")
    nasa_units_var = tk.StringVar(value="—")
    nasa_details_var = tk.StringVar(value="")
    nasa_results_panel = tk.Frame(
        nasa_card, bg=colors["surface_alt"], padx=14, pady=10
    )
    nasa_results_panel.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    nasa_results_panel.columnconfigure(0, weight=1)
    nasa_results_panel.columnconfigure(1, weight=1)
    tk.Label(
        nasa_results_panel,
        text=tr("HASIL HITUNG"),
        bg=colors["surface_alt"],
        fg=colors["muted"],
        font=("Segoe UI", 8, "bold"),
        anchor="w",
    ).grid(row=0, column=0, sticky="w")
    tk.Label(
        nasa_results_panel,
        text=tr("F_int  ·  K_scale = 1.000.000"),
        bg=colors["surface_alt"],
        fg=colors["muted"],
        font=("Segoe UI", 8, "bold"),
        anchor="w",
    ).grid(row=0, column=1, sticky="w")
    tk.Label(
        nasa_results_panel,
        textvariable=nasa_force_var,
        bg=colors["surface_alt"],
        fg=colors["text"],
        font=("Segoe UI", 22, "bold"),
        anchor="w",
    ).grid(row=1, column=0, sticky="w")
    tk.Label(
        nasa_results_panel,
        textvariable=nasa_units_var,
        bg=colors["surface_alt"],
        fg=colors["accent"],
        font=("Segoe UI", 16, "bold"),
        anchor="w",
    ).grid(row=1, column=1, sticky="w")
    tk.Label(
        nasa_results_panel,
        textvariable=nasa_details_var,
        bg=colors["surface_alt"],
        fg=colors["muted"],
        font=("Segoe UI", 9),
        justify="left",
        anchor="w",
        wraplength=760,
    ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0))

    nasa_tree_style = "Astrolab.Treeview"
    style.configure(
        nasa_tree_style,
        background=colors["input"],
        fieldbackground=colors["input"],
        foreground=colors["text"],
        rowheight=28,
        borderwidth=0,
    )
    style.configure(
        "Astrolab.Treeview.Heading",
        background=colors["surface_alt"],
        foreground=colors["muted"],
        font=("Segoe UI", 9, "bold"),
        padding=(8, 7),
    )
    style.map(
        nasa_tree_style,
        background=[("selected", colors["accent_dark"])],
        foreground=[("selected", colors["accent"])],
    )
    nasa_tree_columns = {
        "case": ("Run / sampel", 260, "w"),
        "time": ("Waktu (s)", 75, "e"),
        "flow": ("Aliran (cm/s)", 90, "e"),
        "tc1": ("TC1 (°C)", 80, "e"),
        "radiometer": ("Radiometer (W/m²)", 95, "e"),
        "ignition": ("Ignition", 70, "center"),
        "formula": ("F skenario (N)", 115, "e"),
        "status": ("Status api", 95, "center"),
    }
    nasa_tree = ttk.Treeview(
        nasa_card,
        columns=tuple(nasa_tree_columns),
        show="headings",
        height=5,
        style=nasa_tree_style,
        selectmode="browse",
    )
    for column, (heading, width, anchor) in nasa_tree_columns.items():
        nasa_tree.heading(column, text=tr(heading))
        nasa_tree.column(column, width=width, minwidth=55, anchor=anchor, stretch=True)
    nasa_tree.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
    nasa_tree_headings = {column: values[0] for column, values in nasa_tree_columns.items()}

    def finite_number_or_none(value: object) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if isfinite(parsed) else None

    def refresh_nasa_table() -> None:
        nasa_tree.delete(*nasa_tree.get_children())
        for row in nasa_results:
            formula_force = finite_number_or_none(row["F_formula_Newton"])
            status_label = row.get("fire_status", row["Status"])
            nasa_tree.insert(
                "",
                "end",
                iid=f"nasa-{len(nasa_tree.get_children())}",
                values=(
                    row["test_id"],
                    f'{float(row["TIME_SEC"]):,.1f}' if "TIME_SEC" in row else "—",
                    f'{float(row.get("FLOW_VEL_CMS", row["airflow_speed_cms"])):,.2f}',
                    f'{float(row["TC1_TEMP_C"]):,.1f}' if "TC1_TEMP_C" in row else "—",
                    f'{float(row["RADIOMETER_WM2"]):,.1f}' if "RADIOMETER_WM2" in row else "—",
                    str(row["IGNITION_STATE"]) if "IGNITION_STATE" in row else "—",
                    "—" if formula_force is None else f"{formula_force:,.3f}",
                    tr(str(status_label)),
                ),
            )

    def select_nasa_case(_event=None) -> None:
        selected_name = nasa_case_var.get()
        selected = next(
            (row for row in nasa_results if row["test_id"] == selected_name), None
        )
        if selected is None:
            return
        matching_item = next(
            (
                item_id
                for item_id in nasa_tree.get_children()
                if nasa_tree.item(item_id, "values")[0] == selected_name
            ),
            None,
        )
        if (
            matching_item is not None
            and nasa_tree.selection() != (matching_item,)
        ):
            nasa_tree.selection_set(matching_item)
            nasa_tree.see(matching_item)
        force = finite_number_or_none(selected["F_formula_Newton"])
        units = finite_number_or_none(selected["F_formula_int_Units"])
        formula_force = force
        if selected["Status"] == "CRITICAL":
            nasa_force_var.set(tr("CRITICAL: Singularity Detected"))
            nasa_units_var.set("—")
        elif force is None or units is None:
            nasa_force_var.set("—")
            nasa_units_var.set(tr("No previous sample interval"))
        else:
            nasa_force_var.set(f"{force:,.3f} N")
            nasa_units_var.set(f"{int(units):,} {tr('unit')}")
        if selected.get("data_kind") == "timeseries":
            nasa_details_var.set(
                tr(
                    "{investigation} · {run} · t={time:.1f} s · Δt={duration}\n"
                    "Tekanan={pressure:.2f} kPa · O₂={oxygen:.2f}% · CO₂={co2:.0f} ppm · "
                    "aliran={flow:.2f} cm/s · TC1={tc1:.1f}°C · TC2={tc2:.1f}°C · "
                    "radiometer={radiometer:.1f} W/m² · penyalaan={ignition}\n"
                    "Asumsi DA²={da2} · Cd={cd:.4f} · A={area:.2f} · "
                    "status api={fire_status}: {diagnostic}\n"
                    "Formula literal v2.5={formula} N; hasil skenario, bukan pengukuran NASA."
                ).format(
                    investigation=selected["INVESTIGATION_ID"],
                    run=selected["TEST_RUN_CODE"],
                    time=float(selected["TIME_SEC"]),
                    duration=(
                        "—"
                        if finite_number_or_none(selected["burn_duration_sec"]) is None
                        else f'{float(selected["burn_duration_sec"]):.1f} s'
                    ),
                    pressure=float(selected["PRESSURE_KPA"]),
                    oxygen=float(selected["O2_CONC_PCT"]),
                    co2=float(selected["CO2_CONC_PPM"]),
                    flow=float(selected["FLOW_VEL_CMS"]),
                    tc1=float(selected["TC1_TEMP_C"]),
                    tc2=float(selected["TC2_TEMP_C"]),
                    radiometer=float(selected["RADIOMETER_WM2"]),
                    ignition=int(selected["IGNITION_STATE"]),
                    da2=(
                        "—"
                        if finite_number_or_none(selected["DA2_units"]) is None
                        else f'{float(selected["DA2_units"]):.3f}'
                    ),
                    cd=float(selected["Cd"]),
                    area=float(selected["area"]),
                    fire_status=tr(str(selected["fire_status"])),
                    diagnostic=tr(str(selected["fire_diagnostic"])),
                    formula=(
                        "—"
                        if formula_force is None
                        else f"{float(formula_force):,.3f}"
                    ),
                )
            )
        else:
            nasa_details_var.set(
                tr(
                    "DA² {da2:.3f} · Δt {duration:.3f} s · Cd {cd:.3f} · "
                    "A {area:.3f} · rho relatif {rho:.3f}\n"
                    "Hasil formula: {formula} N"
                ).format(
                    da2=float(selected["DA2_units"]),
                    duration=float(selected["burn_duration_sec"]),
                    cd=float(selected["Cd"]),
                    area=float(selected["area"]),
                    rho=float(selected["rho_weight_relative_to_21pct"]),
                    formula="—" if formula_force is None else f"{float(formula_force):,.3f}",
                )
            )

    def load_nasa_csv() -> None:
        from tkinter import filedialog, messagebox

        selected_path = filedialog.askopenfilename(
            parent=root,
            title=tr("Pilih file CSV"),
            filetypes=((tr("File CSV"), "*.csv"), (tr("Semua file"), "*.*")),
        )
        if not selected_path:
            return
        try:
            frame = hitung_simulasi_integrasi_nasa(
                selected_path,
                da2_scenario=float(nasa_da2_var.get()),
            )
            nasa_results[:] = frame.to_dict(orient="records")
            nasa_source_state["path"] = Path(selected_path)
        except (OSError, UnicodeError, ValueError, ImportError) as error:
            nasa_upload_message.set(f"{tr('CSV gagal diproses: ')}{error}")
            nasa_upload_label.configure(fg=colors["danger"])
            messagebox.showerror(tr("CSV tidak dapat dibaca"), str(error), parent=root)
            return
        case_ids = [str(row["test_id"]) for row in nasa_results]
        nasa_case_select.configure(values=case_ids)
        nasa_case_var.set(case_ids[0])
        fire_case_select.configure(values=case_ids)
        fire_case_var.set(case_ids[0])
        refresh_nasa_table()
        nasa_upload_message.set(
            tr("Berhasil memuat {count} sampel dari {filename}.").format(
                count=len(case_ids), filename=Path(selected_path).name
            )
        )
        nasa_upload_label.configure(fg=colors["accent"])
        select_nasa_case()
        select_fire_case()

    def select_nasa_tree_row(_event=None) -> None:
        selected_items = nasa_tree.selection()
        if not selected_items:
            return
        selected_values = nasa_tree.item(selected_items[0], "values")
        if selected_values:
            nasa_case_var.set(str(selected_values[0]))
        select_nasa_case()

    nasa_tree.bind("<<TreeviewSelect>>", select_nasa_tree_row)
    nasa_results[:] = hitung_simulasi_dataset_contoh(
        da2_scenario=float(nasa_da2_var.get())
    ).to_dict(orient="records")
    nasa_case_ids = [str(row["test_id"]) for row in nasa_results]
    nasa_case_select.configure(values=nasa_case_ids)
    nasa_case_var.set(nasa_case_ids[0])
    refresh_nasa_table()
    nasa_upload_message.set(
        tr("Contoh deret waktu: {count} sampel sensor dimuat.").format(
            count=len(nasa_case_ids)
        )
    )
    nasa_upload_label.configure(fg=colors["muted"])
    select_nasa_case()

    nasa_upload_button = tk.Button(
        nasa_card,
        text=        tr("Impor CSV"),
        command=load_nasa_csv,
        bg=colors["accent_dark"],
        fg=colors["accent"],
        activebackground=colors["border"],
        relief="flat",
        padx=12,
        pady=6,
        cursor="hand2",
    )
    nasa_upload_button.grid(row=3, column=1, sticky="e", pady=(0, 8))
    nasa_case_select.bind("<<ComboboxSelected>>", select_nasa_case)

    def recalculate_nasa_scenario() -> None:
        from tkinter import messagebox

        try:
            da2_scenario = float(nasa_da2_var.get())
            source_path = nasa_source_state["path"]
            frame = (
                hitung_simulasi_integrasi_nasa(
                    source_path, da2_scenario=da2_scenario
                )
                if source_path is not None
                else hitung_simulasi_dataset_contoh(
                    da2_scenario=da2_scenario
                )
            )
        except (OSError, UnicodeError, ValueError, ImportError) as error:
            nasa_upload_message.set(f"{tr('Perhitungan skenario gagal: ')}{error}")
            nasa_upload_label.configure(fg=colors["danger"])
            messagebox.showerror(
                tr("Skenario NASA tidak dapat dihitung"), str(error), parent=root
            )
            return
        nasa_results[:] = frame.to_dict(orient="records")
        case_ids = [str(row["test_id"]) for row in nasa_results]
        nasa_case_select.configure(values=case_ids)
        fire_case_select.configure(values=case_ids)
        if case_ids:
            nasa_case_var.set(case_ids[0])
            fire_case_var.set(case_ids[0])
        refresh_nasa_table()
        nasa_upload_message.set(
            tr("Dihitung ulang {count} sampel dengan asumsi DA² {da2}.").format(
                count=len(case_ids), da2=da2_scenario
            )
        )
        nasa_upload_label.configure(fg=colors["accent"])
        select_nasa_case()
        select_fire_case()

    nasa_recalculate_button = tk.Button(
        nasa_da2_frame,
        text=tr("Terapkan"),
        command=recalculate_nasa_scenario,
        bg=colors["accent_dark"],
        fg=colors["accent"],
        activebackground=colors["border"],
        relief="flat",
        padx=8,
        pady=4,
        cursor="hand2",
    )
    nasa_recalculate_button.pack(side="left", padx=(5, 0))

    fire_page.columnconfigure(0, weight=1)
    fire_page.rowconfigure(0, weight=1)
    fire_card = card(fire_page, padx=18, pady=14)
    fire_card.grid(row=0, column=0, sticky="nsew")
    fire_card.columnconfigure(0, weight=1)
    fire_card.columnconfigure(1, weight=1)
    fire_card.rowconfigure(3, weight=1)

    fire_header = tk.Frame(fire_card, bg=colors["surface"])
    fire_header.grid(row=0, column=0, columnspan=2, sticky="ew")
    label(fire_header, "Panel Fire Safety", size=14, weight="bold").pack(
        side="left"
    )
    label(fire_header, "DEMO · sensor heuristics", size=8, color=colors["muted"]).pack(
        side="right"
    )
    label(
        fire_card,
        "Pemeriksaan input dan visual api sferis untuk demonstrasi. Bukan sensor nyata, "
        "batas keselamatan penerbangan, atau model Saffire tervalidasi.",
        size=9,
        color=colors["muted"],
    ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 10))

    fire_control_row = tk.Frame(fire_card, bg=colors["surface"])
    fire_control_row.grid(
        row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10)
    )
    fire_case_var = tk.StringVar(value="")
    fire_case_select = ttk.Combobox(
        fire_control_row,
        textvariable=fire_case_var,
        state="readonly",
        style="Astrolab.TCombobox",
        width=24,
    )
    fire_case_select.pack(side="left", padx=(0, 8))
    fire_stress_summary = tk.StringVar(value=tr("Pilih skenario untuk memeriksa input."))
    fire_stress_label = tk.Label(
        fire_control_row,
        textvariable=fire_stress_summary,
        bg=colors["surface"],
        fg=colors["muted"],
        font=("Segoe UI", 8),
        anchor="e",
        justify="right",
        wraplength=320,
    )
    fire_stress_label.pack(side="right", fill="x", expand=True, padx=(8, 0))

    fire_visual_card = tk.Frame(
        fire_card,
        bg=colors["input"],
        highlightbackground=colors["border"],
        highlightthickness=1,
        padx=12,
        pady=10,
    )
    fire_visual_card.grid(
        row=3, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8)
    )
    fire_visual_card.columnconfigure(0, weight=1)
    fire_visual_card.rowconfigure(1, weight=1)
    label(
        fire_visual_card,
        "Api sferis · visualisasi mikrogravitasi",
        size=10,
        weight="bold",
    ).grid(row=0, column=0, sticky="w")
    fire_canvas = tk.Canvas(
        fire_visual_card,
        height=250,
        bg=colors["input"],
        highlightthickness=0,
        borderwidth=0,
    )
    fire_canvas.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

    fire_monitor = tk.Frame(
        fire_card,
        bg=colors["surface"],
        highlightbackground=colors["border"],
        highlightthickness=1,
        padx=14,
        pady=12,
    )
    fire_monitor.grid(
        row=3, column=1, sticky="nsew", padx=(6, 0), pady=(0, 8)
    )
    fire_monitor.columnconfigure(0, weight=1)
    fire_status_var = tk.StringVar(value="●  CRITICAL")
    fire_status_badge = tk.Label(
        fire_monitor,
        textvariable=fire_status_var,
        bg="#3b252c",
        fg=colors["danger"],
        padx=10,
        pady=6,
        font=("Segoe UI", 10, "bold"),
        anchor="w",
    )
    fire_status_badge.grid(row=0, column=0, sticky="w")
    fire_force_var = tk.StringVar(value="—")
    tk.Label(
        fire_monitor,
        text=tr("Gaya formula"),
        bg=colors["surface"],
        fg=colors["muted"],
        font=("Segoe UI", 8),
    ).grid(row=1, column=0, sticky="w", pady=(12, 1))
    tk.Label(
        fire_monitor,
        textvariable=fire_force_var,
        bg=colors["surface"],
        fg=colors["text"],
        font=("Segoe UI", 24, "bold"),
        anchor="w",
    ).grid(row=2, column=0, sticky="w")
    fire_units_var = tk.StringVar(value="—")
    tk.Label(
        fire_monitor,
        textvariable=fire_units_var,
        bg=colors["surface"],
        fg=colors["accent"],
        font=("Segoe UI", 12, "bold"),
        anchor="w",
    ).grid(row=3, column=0, sticky="w", pady=(0, 8))
    fire_sensor_var = tk.StringVar(value="")
    tk.Label(
        fire_monitor,
        textvariable=fire_sensor_var,
        bg=colors["surface"],
        fg=colors["muted"],
        font=("Segoe UI", 8),
        justify="left",
        anchor="w",
        wraplength=370,
    ).grid(row=4, column=0, sticky="ew", pady=(0, 8))
    fire_diagnostics_var = tk.StringVar(value="")
    fire_diagnostics_label = tk.Label(
        fire_monitor,
        textvariable=fire_diagnostics_var,
        bg=colors["surface"],
        fg=colors["muted"],
        font=("Segoe UI", 9),
        justify="left",
        anchor="nw",
        wraplength=370,
    )
    fire_diagnostics_label.grid(row=5, column=0, sticky="nsew")
    fire_provenance_var = tk.StringVar(value="")
    tk.Label(
        fire_monitor,
        textvariable=fire_provenance_var,
        bg=colors["surface"],
        fg=colors["muted"],
        font=("Segoe UI", 8),
        justify="left",
        anchor="w",
        wraplength=370,
    ).grid(row=6, column=0, sticky="ew", pady=(10, 0))

    fire_state: dict[str, object] = {
        "status": "CRITICAL",
        "phase": 0.0,
        "after_id": None,
    }

    def draw_fire_sphere() -> None:
        width = max(fire_canvas.winfo_width(), 240)
        height = max(fire_canvas.winfo_height(), 180)
        center_x, center_y = width * 0.5, height * 0.48
        phase = float(fire_state["phase"])
        pulse = 0.5 + 0.5 * np.sin(phase)
        outer_radius = min(width, height) * (0.25 + 0.012 * pulse)
        critical = fire_state["status"] == "CRITICAL"
        palette = (
            ("#522b33", "#823841", "#b74740", "#e36042", "#ff8c43", "#ffd36b", "#fff1bd")
            if critical
            else ("#403049", "#6a3b54", "#a94e57", "#ed7148", "#ffae50", "#ffe17a", "#fff5ce")
        )
        fire_canvas.delete("all")
        fire_canvas.create_oval(
            center_x - outer_radius * 1.4,
            center_y - outer_radius * 1.4,
            center_x + outer_radius * 1.4,
            center_y + outer_radius * 1.4,
            outline="#bd6271" if critical else "#66cbb3",
            width=1,
            dash=(3, 5),
        )
        for layer, color in enumerate(palette):
            radius = outer_radius * (1.0 - layer * 0.105)
            fire_canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                fill=color,
                outline="",
            )
        fire_canvas.create_text(
            center_x,
            min(height - 16, center_y + outer_radius * 1.62),
            text=tr("Visual demo · bukan hasil kamera atau sensor"),
            fill=colors["muted"],
            font=("Segoe UI", 8),
        )

    def update_fire_safety(
        measurements: dict[str, object],
        provenance: str,
        *,
        stress_elapsed_ms: float | None = None,
    ) -> None:
        assessment = assess_fire_safety(measurements)
        critical = assessment.status == "CRITICAL"
        fire_state["status"] = assessment.status
        fire_status_var.set(
            tr("●  CRITICAL") if critical else tr("●  STABLE")
        )
        fire_status_badge.configure(
            bg="#3b252c" if critical else colors["accent_dark"],
            fg=colors["danger"] if critical else colors["accent"],
        )
        if assessment.force_newton is None or assessment.force_int_units is None:
            fire_force_var.set("—")
            fire_units_var.set("—")
        else:
            fire_force_var.set(f"{assessment.force_newton:,.3f} N")
            fire_units_var.set(f"{assessment.force_int_units:,} {tr('unit')}")
        diameter = float(measurements["flame_diameter_mm"])
        duration = float(measurements["burn_duration_sec"])
        airflow = float(measurements["airflow_speed_cms"])
        oxygen_text = (
            "—"
            if assessment.oxygen_percent is None
            else f"{assessment.oxygen_percent:.2f}%"
        )
        fire_sensor_var.set(
            tr(
                "Ø {diameter:.2f} mm · Δt {duration:.2f} s\n"
                "Airflow {airflow:.2f} cm/s · O₂ {oxygen} · {latency:.3f} ms"
            ).format(
                diameter=diameter,
                duration=duration,
                airflow=airflow,
                oxygen=oxygen_text,
                latency=assessment.latency_ms,
            )
        )
        fire_diagnostics_var.set(
            "\n".join(tr(item) for item in assessment.diagnostics)
        )
        fire_diagnostics_label.configure(
            fg=colors["danger"] if critical else colors["accent"]
        )
        provenance_labels = {
            "built-in": "Contoh proyek; bukan data NASA terverifikasi.",
            "user-csv": "CSV pengguna; belum diverifikasi secara independen.",
            "stress": "Profil stress sintetis; bukan pengukuran eksperimen.",
        }
        fire_provenance_var.set(
            tr(provenance_labels[provenance])
        )
        if stress_elapsed_ms is not None:
            fire_stress_summary.set(
                tr("Stress {checks} cek · {elapsed:.2f} ms · {status}").format(
                    checks=250,
                    elapsed=stress_elapsed_ms,
                    status=tr(assessment.status),
                )
            )
        else:
            fire_stress_summary.set(tr("Pemeriksaan input selesai."))
        draw_fire_sphere()

    def select_fire_case(_event=None) -> None:
        selected_name = fire_case_var.get()
        selected = next(
            (row for row in nasa_results if row["test_id"] == selected_name), None
        )
        if selected is None:
            return
        if selected.get("data_kind") == "timeseries":
            raw_status = str(selected["fire_status"])
            fire_state["status"] = (
                "CRITICAL" if raw_status in ("CRITICAL", "WATCH") else "STABLE"
            )
            status_color = {
                "IDLE": colors["muted"],
                "STABLE": colors["accent"],
                "WATCH": colors["orange"],
                "CRITICAL": colors["danger"],
            }[raw_status]
            fire_status_var.set(tr(f"●  {raw_status}"))
            fire_status_badge.configure(
                bg=(
                    "#3b252c"
                    if raw_status == "CRITICAL"
                    else colors["accent_dark"]
                    if raw_status == "STABLE"
                    else colors["surface_alt"]
                ),
                fg=status_color,
            )
            force = finite_number_or_none(selected["F_formula_Newton"])
            units = finite_number_or_none(selected["F_formula_int_Units"])
            fire_force_var.set(
                "—"
                if force is None
                else f"{force:,.3f} N"
            )
            fire_units_var.set(
                tr("No previous sample interval")
                if units is None
                else f"{int(units):,} {tr('unit')}"
            )
            fire_sensor_var.set(
                tr(
                    "Waktu={time:.1f} s · Δt={interval} · tekanan={pressure:.2f} kPa\n"
                    "O₂={oxygen:.2f}% · CO₂={co2:.0f} ppm · aliran={flow:.2f} cm/s\n"
                    "TC1={tc1:.1f}°C · TC2={tc2:.1f}°C · radiometer={radiometer:.1f} W/m²\n"
                    "Penyalaan={ignition} · asumsi DA²={da2} · Cd={cd:.4f}"
                ).format(
                    time=float(selected["TIME_SEC"]),
                    interval=(
                        "—"
                        if finite_number_or_none(selected["burn_duration_sec"]) is None
                        else f'{float(selected["burn_duration_sec"]):.1f} s'
                    ),
                    pressure=float(selected["PRESSURE_KPA"]),
                    oxygen=float(selected["O2_CONC_PCT"]),
                    co2=float(selected["CO2_CONC_PPM"]),
                    flow=float(selected["FLOW_VEL_CMS"]),
                    tc1=float(selected["TC1_TEMP_C"]),
                    tc2=float(selected["TC2_TEMP_C"]),
                    radiometer=float(selected["RADIOMETER_WM2"]),
                    ignition=int(selected["IGNITION_STATE"]),
                    da2=(
                        "—"
                        if finite_number_or_none(selected["DA2_units"]) is None
                        else f'{float(selected["DA2_units"]):.3f}'
                    ),
                    cd=float(selected["Cd"]),
                )
            )
            fire_diagnostics_var.set(tr(str(selected["fire_diagnostic"])))
            fire_diagnostics_label.configure(fg=status_color)
            fire_provenance_var.set(
                tr("Deret waktu dari pengguna; asal data belum diverifikasi independen.")
            )
            fire_stress_summary.set(
                tr("Sampel input {sample} / {run}").format(
                    sample=f'{float(selected["TIME_SEC"]):g} s',
                    run=selected["TEST_RUN_CODE"],
                )
            )
            draw_fire_sphere()
            return
        measurements = {
            "flame_diameter_mm": selected["flame_diameter_mm"],
            "burn_duration_sec": selected["burn_duration_sec"],
            "airflow_speed_cms": selected["airflow_speed_cms"],
            "oxygen_concentration": selected["oxygen_concentration_percent"],
        }
        provenance = (
            "user-csv"
            if str(selected["data_provenance"]).startswith("user-provided")
            else "built-in"
        )
        update_fire_safety(measurements, provenance)

    def run_fire_stress_test() -> None:
        from time import perf_counter

        measurements = stress_test_measurements()
        started = perf_counter()
        assessment = None
        for _ in range(250):
            assessment = assess_fire_safety(measurements)
        elapsed_ms = (perf_counter() - started) * 1000
        if assessment is None:
            raise RuntimeError("The fixed stress-test batch produced no assessment.")
        update_fire_safety(
            measurements, "stress", stress_elapsed_ms=elapsed_ms
        )

    fire_stress_button = tk.Button(
        fire_control_row,
        text=tr("Uji stress sintetis"),
        command=run_fire_stress_test,
        bg=colors["surface_alt"],
        fg=colors["text"],
        activebackground=colors["border"],
        relief="flat",
        padx=10,
        pady=6,
        cursor="hand2",
    )
    fire_stress_button.pack(side="left")
    fire_case_select.bind("<<ComboboxSelected>>", select_fire_case)
    fire_case_ids = [str(row["test_id"]) for row in nasa_results]
    fire_case_select.configure(values=fire_case_ids)
    if fire_case_ids:
        fire_case_var.set(fire_case_ids[0])
        select_fire_case()

    graph_card = card(home_scroll_frame, padx=18, pady=14)
    graph_card.grid(row=1, column=0, sticky="ew")
    graph_card.columnconfigure(0, weight=1)
    graph_card.columnconfigure(1, weight=1)
    graph_header = tk.Frame(graph_card, bg=colors["surface"])
    graph_header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    label(graph_header, "Profil aliran demonstrasi", size=12, weight="bold").pack(
        side="left"
    )
    label(graph_header, "kurva ilustratif · bukan data penerbangan", size=9, color=colors["muted"]).pack(
        side="right"
    )

    graphs: dict[str, tk.Canvas] = {}
    for column, (key, title, color) in enumerate(
        (
            ("flame", "Ekspansi radius relatif", colors["orange"]),
            ("thrust", "Respons dorong relatif", colors["blue"]),
        )
    ):
        graph_wrap = tk.Frame(graph_card, bg=colors["surface"])
        graph_wrap.grid(row=1, column=column, sticky="ew", padx=(0, 8) if column == 0 else (8, 0))
        graph_wrap.columnconfigure(0, weight=1)
        label(graph_wrap, title, size=9, color=colors["muted"]).grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        canvas = tk.Canvas(
            graph_wrap,
            height=110,
            bg=colors["input"],
            highlightthickness=0,
            borderwidth=0,
        )
        canvas.grid(row=1, column=0, sticky="ew")
        graphs[key] = canvas
        graphs[key].graph_color = color

    animation_buttons: dict[str, tk.Button] = {}
    rocket_card = card(rocket_page, padx=20, pady=18)
    rocket_card.pack(fill="both", expand=True)
    label(rocket_card, "Roket · orbit rendah Bumi", size=15, weight="bold").pack(
        anchor="w"
    )
    label(
        rocket_card,
        "Propagasi gravitasi dua-benda dengan SciPy DOP853. Orbit tidak menggunakan "
        "dorongan formula kustom. Seret grafik untuk memutar sudut pandang 3D.",
        size=9,
        color=colors["muted"],
    ).pack(anchor="w", pady=(5, 12))
    rocket_controls = tk.Frame(rocket_card, bg=colors["surface"])
    rocket_controls.pack(fill="x", pady=(0, 8))
    rocket_input_row = tk.Frame(rocket_controls, bg=colors["surface"])
    rocket_input_row.pack(fill="x")
    rocket_action_row = tk.Frame(rocket_controls, bg=colors["surface"])
    rocket_action_row.pack(fill="x")
    rocket_altitude_var = tk.StringVar(value="400")
    rocket_speed_ratio_var = tk.StringVar(value="1.0")
    rocket_inclination_var = tk.StringVar(value="51.6")
    rocket_revolutions_var = tk.StringVar(value="1")
    rocket_view_var = tk.StringVar(value="3D")
    rocket_rotation_var = tk.BooleanVar(value=True)

    def add_compact_input(parent, title: str, variable, width: int = 7) -> None:
        group = tk.Frame(parent, bg=colors["surface"])
        group.pack(side="left", padx=(0, 12))
        label(group, title, size=8, color=colors["muted"]).pack(anchor="w")
        tk.Entry(
            group,
            textvariable=variable,
            width=width,
            bg=colors["input"],
            fg=colors["text"],
            insertbackground=colors["accent"],
            relief="flat",
            justify="right",
        ).pack(ipady=4)

    add_compact_input(rocket_input_row, "Ketinggian (km)", rocket_altitude_var)
    add_compact_input(rocket_input_row, "Rasio kecepatan", rocket_speed_ratio_var)
    add_compact_input(rocket_input_row, "Inklinasi (°)", rocket_inclination_var)
    add_compact_input(rocket_input_row, "Orbit", rocket_revolutions_var, 5)
    label(rocket_action_row, "Tampilan", size=8, color=colors["muted"]).pack(
        side="left", padx=(0, 5), pady=(12, 0)
    )
    rocket_view_select = ttk.Combobox(
        rocket_action_row,
        textvariable=rocket_view_var,
        values=("2D", "3D"),
        state="readonly",
        style="Astrolab.TCombobox",
        width=4,
    )
    rocket_view_select.pack(side="left", padx=(0, 10), pady=(12, 0))
    ttk.Checkbutton(
        rocket_action_row,
        text=tr("Rotasi lembut saat diputar"),
        variable=rocket_rotation_var,
        style="Astrolab.TCheckbutton",
    ).pack(side="left", padx=(0, 8), pady=(10, 0))
    rocket_summary = tk.StringVar(value="Tekan Simulasikan untuk propagasi orbit.")
    tk.Label(
        rocket_card,
        textvariable=rocket_summary,
        bg=colors["surface"],
        fg=colors["accent"],
        font=("Segoe UI", 9, "bold"),
        anchor="w",
    ).pack(fill="x", pady=(4, 8))
    tk.Button(
        rocket_action_row,
        text=tr("Simulasikan orbit"),
        command=lambda: refresh_rocket(),
        bg=colors["accent_dark"],
        fg=colors["accent"],
        activebackground=colors["border"],
        relief="flat",
        padx=12,
        pady=6,
        cursor="hand2",
    ).pack(side="right", padx=(4, 0), pady=(10, 0))
    animation_buttons["rocket"] = tk.Button(
        rocket_action_row,
        text=tr("Putar simulasi"),
        command=lambda: toggle_animation("rocket"),
        bg=colors["surface_alt"],
        fg=colors["text"],
        activebackground=colors["border"],
        relief="flat",
        padx=10,
        pady=6,
        cursor="hand2",
    )
    animation_buttons["rocket"].pack(side="right", padx=(4, 0), pady=(10, 0))

    rocket_figure = Figure(figsize=(8, 5.3), dpi=78, facecolor=colors["surface"])
    rocket_axes = rocket_figure.add_subplot(111, projection="3d")
    rocket_figure.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
    rocket_canvas = FigureCanvasTkAgg(rocket_figure, master=rocket_card)
    rocket_canvas.get_tk_widget().pack(fill="both", expand=True)

    simulation_data: dict[str, np.ndarray | None] = {"rocket": None}
    simulation_markers: dict[str, object | None] = {"rocket": None}
    animation_active = {"rocket": False}
    animation_after_ids: dict[str, str | None] = {"rocket": None}
    animation_indices = {"rocket": 0}
    earth_texture = load_earth_texture()
    rocket_scene_state: dict[str, object | None] = {
        "earth_image": None,
        "front_orbit": None,
        "back_orbit": None,
    }
    rocket_rotation = {"angle": 0.0}
    calculation_state: dict[str, object] = {
        "result": None,
        "status_text": "●  STABLE",
        "status_background": colors["accent_dark"],
        "status_foreground": colors["accent"],
        "force_text": "—",
        "units_text": "—",
        "message": "Nilai stabil dihitung dengan Decimal.",
        "message_color": colors["muted"],
    }

    def stop_animation(kind: str) -> None:
        animation_active[kind] = False
        after_id = animation_after_ids[kind]
        if after_id is not None:
            root.after_cancel(after_id)
            animation_after_ids[kind] = None
        button = animation_buttons.get(kind)
        if button is not None:
            button.configure(text=tr("Putar simulasi"))

    def current_force() -> float | None:
        result = calculation_state["result"]
        if not isinstance(result, dict) or result.get("Status") != "STABLE":
            return None
        force = result.get("F_Newton")
        return force if isinstance(force, float) else None

    def style_3d_axes(axes, title: str) -> None:
        axes.set_facecolor(colors["input"])
        axes.set_title(title, color=colors["text"], fontsize=10, pad=5)
        axes.tick_params(colors=colors["muted"], labelsize=7, pad=0)
        axes.xaxis.label.set_color(colors["muted"])
        axes.yaxis.label.set_color(colors["muted"])
        axes.zaxis.label.set_color(colors["muted"])
        for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
            axis.pane.set_facecolor(colors["input"])
            axis.pane.set_edgecolor(colors["border"])
            axis._axinfo["grid"]["color"] = colors["border"]

    def draw_earth(axes, altitude_km: float, positions: np.ndarray) -> None:
        sphere_x, sphere_y, sphere_z, surface_rgba, cloud_rgba = (
            earth_sphere_mesh(
                earth_texture,
                radius_km=EARTH_RADIUS_KM,
                longitude_samples=49,
                latitude_samples=25,
            )
        )
        for shell_scale, shell_alpha in ((1.018, 0.035), (1.008, 0.075)):
            axes.plot_surface(
                sphere_x * shell_scale,
                sphere_y * shell_scale,
                sphere_z * shell_scale,
                color="#58bfff",
                alpha=shell_alpha,
                linewidth=0,
                antialiased=False,
                shade=False,
                zorder=1,
            )
        axes.plot_surface(
            sphere_x,
            sphere_y,
            sphere_z,
            facecolors=surface_rgba,
            linewidth=0,
            antialiased=False,
            shade=False,
            zorder=2,
        )
        axes.plot_surface(
            sphere_x * 1.003,
            sphere_y * 1.003,
            sphere_z * 1.003,
            facecolors=cloud_rgba,
            linewidth=0,
            antialiased=False,
            shade=False,
            zorder=3,
        )
        orbit_extent = float(np.max(np.linalg.norm(positions, axis=1)))
        limit = orbit_extent + max(250.0, altitude_km * 0.2)
        axes.set_xlim(-limit, limit)
        axes.set_ylim(-limit, limit)
        axes.set_zlim(-limit, limit)
        axes.set_box_aspect((1, 1, 1))
        axes.set_xlabel("X (km)", labelpad=-2)
        axes.set_ylabel("Y (km)", labelpad=-2)
        axes.set_zlabel("Z (km)", labelpad=-2)
        style_3d_axes(axes, tr("Orbit 3D · inersial terpusat di Bumi"))

    def update_orbit_segments(positions: np.ndarray) -> None:
        front_line = rocket_scene_state["front_orbit"]
        back_line = rocket_scene_state["back_orbit"]
        if front_line is None or back_line is None or len(positions) < 2:
            return
        if rocket_view_var.get() == "3D":
            elevation = np.deg2rad(rocket_axes.elev)
            azimuth = np.deg2rad(rocket_axes.azim)
            camera_direction = np.asarray(
                (
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                )
            )
        else:
            camera_direction = np.asarray((0.0, 0.0, 1.0))
        segment_visibility = visibility_mask(positions, camera_direction)
        front_positions = np.full(positions.shape, np.nan, dtype=np.float64)
        back_positions = np.full(positions.shape, np.nan, dtype=np.float64)
        front_positions[:-1][segment_visibility] = positions[:-1][segment_visibility]
        front_positions[1:][segment_visibility] = positions[1:][segment_visibility]
        back_positions[:-1][~segment_visibility] = positions[:-1][~segment_visibility]
        back_positions[1:][~segment_visibility] = positions[1:][~segment_visibility]
        if rocket_view_var.get() == "3D":
            front_line.set_data_3d(
                front_positions[:, 0], front_positions[:, 1], front_positions[:, 2]
            )
            back_line.set_data_3d(
                back_positions[:, 0], back_positions[:, 1], back_positions[:, 2]
            )
        else:
            front_line.set_data(front_positions[:, 0], front_positions[:, 1])
            back_line.set_data(back_positions[:, 0], back_positions[:, 1])

    def refresh_rocket() -> None:
        nonlocal rocket_axes
        stop_animation("rocket")
        try:
            altitude = float(rocket_altitude_var.get())
            speed_ratio = float(rocket_speed_ratio_var.get())
            inclination = float(rocket_inclination_var.get())
            revolutions = float(rocket_revolutions_var.get())
            times, positions = simulate_earth_orbit(
                altitude,
                speed_ratio,
                inclination,
                revolutions,
                samples=420,
            )
        except (ValueError, RuntimeError) as error:
            rocket_summary.set(f"{tr('Input/simulasi tidak valid: ')}{error}")
            rocket_figure.clear()
            rocket_axes = rocket_figure.add_subplot(
                111, projection="3d" if rocket_view_var.get() == "3D" else None
            )
            rocket_axes.text2D(
                0.1, 0.5, str(error), transform=rocket_axes.transAxes,
                color=colors["danger"],
            ) if rocket_view_var.get() == "3D" else rocket_axes.text(
                0.1, 0.5, str(error), transform=rocket_axes.transAxes,
                color=colors["danger"],
            )
            rocket_canvas.draw_idle()
            return
        radius_km = EARTH_RADIUS_KM + altitude
        semi_major_axis = radius_km / (2 - speed_ratio**2)
        period_s = 2 * np.pi * np.sqrt(
            semi_major_axis**3 / EARTH_MU_KM3_S2
        )
        orbital_speed = np.sqrt(EARTH_MU_KM3_S2 / radius_km) * speed_ratio
        rocket_summary.set(
            tr(
                "Radius orbit {radius:,.0f} km   ·   Kecepatan awal {speed:,.3f} km/s   ·   "
                "Periode {period:,.1f} menit   ·   {samples} titik"
            ).format(
                radius=radius_km,
                speed=orbital_speed,
                period=period_s / 60,
                samples=len(times),
            )
        )
        animation_active["rocket"] = False
        simulation_data["rocket"] = positions
        animation_indices["rocket"] = 0
        rocket_rotation["angle"] = 0.0
        rocket_scene_state.update(
            earth_image=None, front_orbit=None, back_orbit=None
        )
        rocket_figure.clear()
        rocket_axes = rocket_figure.add_subplot(
            111, projection="3d" if rocket_view_var.get() == "3D" else None
        )
        if rocket_view_var.get() == "3D":
            draw_earth(rocket_axes, altitude, positions)
            rocket_scene_state["back_orbit"], = rocket_axes.plot(
                [], [], [],
                color="#60758d",
                alpha=0.48,
                linewidth=1.35,
                linestyle=(0, (3, 5)),
                label=tr("Orbit sisi belakang"),
            )
            rocket_scene_state["front_orbit"], = rocket_axes.plot(
                [], [], [],
                color=colors["orange"],
                linewidth=1.8,
                linestyle=(0, (4, 3)),
                label=tr("Orbit sisi depan"),
            )
            simulation_markers["rocket"] = rocket_axes.scatter(
                [positions[0, 0]], [positions[0, 1]], [positions[0, 2]],
                color=colors["accent"], s=28, label=tr("Posisi awal"),
            )
        else:
            from matplotlib.patches import Circle

            rocket_axes.add_patch(
                Circle(
                    (0, 0),
                    EARTH_RADIUS_KM * 1.055,
                    color="#278bd1",
                    alpha=0.08,
                    linewidth=0,
                    zorder=0,
                )
            )
            rocket_axes.add_patch(
                Circle(
                    (0, 0),
                    EARTH_RADIUS_KM * 1.018,
                    color="#2679b5",
                    alpha=0.14,
                    linewidth=0,
                    zorder=1,
                )
            )
            rocket_scene_state["earth_image"] = rocket_axes.imshow(
                render_earth_orthographic(
                    earth_texture,
                    size=320,
                    rotation_rad=rocket_rotation["angle"],
                ),
                extent=(
                    -EARTH_RADIUS_KM,
                    EARTH_RADIUS_KM,
                    -EARTH_RADIUS_KM,
                    EARTH_RADIUS_KM,
                ),
                origin="lower",
                interpolation="bilinear",
                zorder=2,
            )
            rocket_scene_state["back_orbit"], = rocket_axes.plot(
                [], [],
                color="#60758d",
                alpha=0.48,
                linewidth=1.35,
                linestyle=(0, (3, 5)),
                label=tr("Orbit sisi belakang"),
                zorder=3,
            )
            rocket_scene_state["front_orbit"], = rocket_axes.plot(
                [], [],
                color=colors["orange"],
                linewidth=1.8,
                linestyle=(0, (4, 3)),
                label=tr("Orbit sisi depan"),
                zorder=4,
            )
            simulation_markers["rocket"], = rocket_axes.plot(
                [positions[0, 0]], [positions[0, 1]], "o",
                color=colors["accent"], markersize=6, label=tr("Posisi awal"),
                zorder=5,
            )
            orbit_limit = radius_km * 1.08
            rocket_axes.set_xlim(-orbit_limit, orbit_limit)
            rocket_axes.set_ylim(-orbit_limit, orbit_limit)
            rocket_axes.set_aspect("equal", adjustable="box")
            rocket_axes.set_xlabel("X (km)", color=colors["muted"])
            rocket_axes.set_ylabel("Y (km)", color=colors["muted"])
            rocket_axes.set_title(tr("Orbit 2D · tampak atas"), color=colors["text"])
            rocket_axes.tick_params(colors=colors["muted"], labelsize=7)
            rocket_axes.set_facecolor(colors["input"])
            rocket_axes.grid(color=colors["border"], linewidth=0.5)
        update_orbit_segments(positions)
        rocket_axes.legend(
            loc="upper left", fontsize=7, facecolor=colors["surface"],
            labelcolor=colors["text"], framealpha=0.8,
        )
        animation_buttons["rocket"].configure(text=tr("Putar simulasi"))
        rocket_canvas.draw_idle()

    def toggle_animation(kind: str) -> None:
        trajectory = simulation_data[kind]
        if trajectory is None:
            refresh_rocket()
            trajectory = simulation_data[kind]
        if trajectory is None:
            return
        if animation_active[kind]:
            stop_animation(kind)
            return
        animation_active[kind] = True
        animation_buttons[kind].configure(text=tr("Jeda simulasi"))
        animate_step(kind)

    def animate_step(kind: str) -> None:
        trajectory = simulation_data[kind]
        marker = simulation_markers[kind]
        if not animation_active[kind] or trajectory is None or marker is None:
            return
        index = animation_indices[kind]
        if rocket_view_var.get() == "3D":
            marker._offsets3d = (
                [trajectory[index, 0]],
                [trajectory[index, 1]],
                [trajectory[index, 2]],
            )
            if rocket_rotation_var.get():
                rocket_axes.view_init(
                    elev=rocket_axes.elev,
                    azim=(rocket_axes.azim + 0.14) % 360,
                )
                update_orbit_segments(trajectory)
        else:
            marker.set_data([trajectory[index, 0]], [trajectory[index, 1]])
            if rocket_rotation_var.get():
                rocket_rotation["angle"] += 0.003
                earth_image = render_earth_orthographic(
                    earth_texture,
                    size=320,
                    rotation_rad=rocket_rotation["angle"],
                )
                earth_artist = rocket_scene_state["earth_image"]
                if earth_artist is not None:
                    earth_artist.set_data(earth_image)
        rocket_canvas.draw_idle()
        animation_indices[kind] = (index + 2) % len(trajectory)
        animation_after_ids[kind] = root.after(80, lambda: animate_step(kind))

    def draw_profile(canvas, kind, current_force):
        canvas.delete("all")
        width = max(canvas.winfo_width(), 220)
        height = max(canvas.winfo_height(), 100)
        pad_x, pad_y = 32, 12
        plot_right, plot_bottom = width - 10, height - 23
        canvas.create_line(
            pad_x, pad_y, pad_x, plot_bottom, fill=colors["border"], width=1
        )
        canvas.create_line(
            pad_x, plot_bottom, plot_right, plot_bottom, fill=colors["border"], width=1
        )
        for fraction in (0.25, 0.5, 0.75):
            y = pad_y + (plot_bottom - pad_y) * fraction
            canvas.create_line(pad_x + 1, y, plot_right, y, fill=colors["border"])

        points = []
        for index in range(51):
            t = index / 50 * 20
            if kind == "flame":
                value = 0.25 + 0.035 * t
                normalized = (value - 0.25) / (1.125 - 0.25)
            else:
                base = 1 + min(abs(current_force), 100) / 100
                value = base + 0.35 * (1 - exp(-t / 4))
                normalized = (value - 1) / 1.8
            x = pad_x + (plot_right - pad_x) * index / 50
            y = plot_bottom - max(0, min(1, normalized)) * (plot_bottom - pad_y)
            points.extend((x, y))
        canvas.create_line(
            *points,
            fill=canvas.graph_color,
            width=2,
            smooth=True,
        )
        canvas.create_text(
            pad_x, height - 8, text="0 s", fill=colors["muted"], anchor="w",
            font=("Segoe UI", 8),
        )
        canvas.create_text(
            plot_right, height - 8, text="20 s", fill=colors["muted"], anchor="e",
            font=("Segoe UI", 8),
        )

    def update_result(*_):
        mode = {
            tr("Formula dinamis awal"): "dynamic",
            tr("Persamaan v2.5 literal"): "v2_5",
            tr("Referensi manual 22 N"): "manual_reference",
        }[mode_var.get()]
        try:
            values = {
                name: engine._decimal(var.get(), name)
                for name, var in parameter_vars.items()
            }
            result = engine.calculate_sector3_status(
                values["DA²"],
                values["delta_t"],
                values["Cd"],
                values["Area"],
                values["Divergensi / grad"],
                mode,
            )
        except ValueError as error:
            calculation_state.update(
                status_text=tr("●  INPUT ERROR"),
                status_background="#3b252c",
                status_foreground=colors["danger"],
                message=str(error),
                message_color=colors["danger"],
            )
            output_message.set(str(error))
            message_label.configure(fg=colors["danger"])
            status_badge.configure(
                text=calculation_state["status_text"],
                bg=calculation_state["status_background"],
                fg=calculation_state["status_foreground"],
            )
            return
        except (ArithmeticError, OverflowError) as error:
            calculation_state.update(
                status_text=tr("●  CRITICAL"),
                status_background="#3b252c",
                status_foreground=colors["danger"],
                force_text="—",
                units_text="—",
                message=f"{tr('Perhitungan gagal: ')}{error}",
                message_color=colors["danger"],
                result=None,
            )
            output_message.set(f"{tr('Perhitungan gagal: ')}{error}")
            message_label.configure(fg=colors["danger"])
            force_value.set(calculation_state["force_text"])
            units_value.set(calculation_state["units_text"])
            status_badge.configure(
                text=calculation_state["status_text"],
                bg=calculation_state["status_background"],
                fg=calculation_state["status_foreground"],
            )
            return

        if result["Status"] == "CRITICAL":
            calculation_state.update(
                result=result,
                status_text=tr("●  CRITICAL"),
                status_background="#3b252c",
                status_foreground=colors["danger"],
                force_text="—",
                units_text="—",
                message=tr(CRITICAL_SINGULARITY),
                message_color=colors["danger"],
            )
        else:
            calculation_state.update(
                result=result,
                status_text=tr("●  STABLE"),
                status_background=colors["accent_dark"],
                status_foreground=colors["accent"],
                force_text=f'{result["F_Newton"]:,.6g} N',
                units_text=f'{result["F_int_Units"]:,} {tr("unit")}',
                message=tr("Output gaya skalar dihitung dengan Decimal."),
                message_color=colors["muted"],
            )

        status_badge.configure(
            text=calculation_state["status_text"],
            bg=calculation_state["status_background"],
            fg=calculation_state["status_foreground"],
        )
        force_value.set(calculation_state["force_text"])
        units_value.set(calculation_state["units_text"])
        output_message.set(calculation_state["message"])
        message_label.configure(fg=calculation_state["message_color"])
        force = current_force()
        if force is not None:
            for key, canvas in graphs.items():
                draw_profile(canvas, key, force)

    for var in parameter_vars.values():
        var.trace_add("write", update_result)
    for canvas in graphs.values():
        canvas.bind(
            "<Configure>",
            lambda event: draw_profile(
                event.widget,
                "flame" if event.widget is graphs["flame"] else "thrust",
                current_force() if current_force() is not None else 1.0,
            ),
        )

    def reset_parameters() -> None:
        for name, value in default_values.items():
            parameter_vars[name].set(value)
        mode_var.set(tr("Formula dinamis awal"))
        rocket_altitude_var.set("400")
        rocket_speed_ratio_var.set("1.0")
        rocket_inclination_var.set("51.6")
        rocket_revolutions_var.set("1")
        update_result()
        refresh_rocket()

    current_page = {"name": "Beranda"}

    def animate_fire_sphere() -> None:
        fire_state["after_id"] = None
        if current_page["name"] != "Fire Safety":
            return
        fire_state["phase"] = (float(fire_state["phase"]) + 0.18) % (2 * np.pi)
        draw_fire_sphere()
        fire_state["after_id"] = root.after(80, animate_fire_sphere)

    def show_page(page_name: str) -> None:
        previous_page = current_page["name"]
        if previous_page != page_name:
            previous_simulation = {
                "Roket vakum": "rocket",
            }.get(previous_page)
            if previous_simulation is not None:
                stop_animation(previous_simulation)
            if previous_page == "Fire Safety":
                fire_after_id = fire_state["after_id"]
                if fire_after_id is not None:
                    root.after_cancel(fire_after_id)
                    fire_state["after_id"] = None
        current_page["name"] = page_name
        pages[page_name].tkraise()
        for button_name, button in nav_buttons.items():
            active = button_name == page_name
            button.configure(
                bg=colors["accent_dark"] if active else colors["surface_alt"],
                fg=colors["accent"] if active else colors["muted"],
            )
        if page_name == "Beranda":
            status_badge.configure(
                text=calculation_state["status_text"],
                bg=calculation_state["status_background"],
                fg=calculation_state["status_foreground"],
            )
            force_value.set(calculation_state["force_text"])
            units_value.set(calculation_state["units_text"])
            output_message.set(calculation_state["message"])
            message_label.configure(fg=calculation_state["message_color"])
        elif page_name == "Roket vakum":
            root.after_idle(refresh_rocket)
        elif page_name == "Fire Safety":
            select_fire_case()
            if fire_state["after_id"] is None:
                fire_state["after_id"] = root.after_idle(animate_fire_sphere)

    nav_buttons: dict[str, tk.Button] = {}
    for page_name in pages:
        button = tk.Button(
            navigation,
            text=page_name,
            command=lambda name=page_name: show_page(name),
            bg=colors["accent_dark"] if page_name == "Beranda" else colors["surface_alt"],
            fg=colors["accent"] if page_name == "Beranda" else colors["muted"],
            activebackground=colors["border"],
            activeforeground=colors["text"],
            relief="flat",
            padx=14,
            pady=7,
            cursor="hand2",
            font=("Segoe UI", 9, "bold"),
        )
        button.pack(side="left", padx=(0, 7))
        nav_buttons[page_name] = button

    reset_button = tk.Button(
        header,
        text="Atur ulang",
        command=reset_parameters,
        bg=colors["surface_alt"],
        fg=colors["text"],
        activebackground=colors["border"],
        activeforeground=colors["text"],
        relief="flat",
        padx=12,
        pady=7,
        cursor="hand2",
        font=("Segoe UI", 9),
    )
    reset_button.grid(row=0, column=2, sticky="e", padx=(10, 0))

    language_options = {
        "id": "id | Indonesia",
        "en": "en | English",
        "es": "es | Español",
    }
    language_var = tk.StringVar(value=language_options["id"])
    language_label = label(header, "Pilih bahasa", size=8, color=colors["muted"])
    language_label.grid(row=0, column=3, sticky="e", padx=(12, 5))
    language_select = ttk.Combobox(
        header,
        textvariable=language_var,
        values=tuple(language_options.values()),
        state="readonly",
        style="Astrolab.TCombobox",
        width=10,
    )
    language_select.grid(row=0, column=4, sticky="e", padx=(0, 2))
    def translate_ui(_event=None) -> None:
        selected_code = language_var.get().split("|", 1)[0].strip()
        if selected_code not in TRANSLATIONS:
            selected_code = "id"
        previous_code = language["code"]
        selected_mode = next(
            (
                source
                for source in (
                    "Formula dinamis awal",
                    "Persamaan v2.5 literal",
                    "Referensi manual 22 N",
                )
                if mode_var.get() == translate(source, previous_code)
            ),
            "Formula dinamis awal",
        )
        language["code"] = selected_code
        for widget, source_text in widget_text_sources.items():
            try:
                widget.configure(text=tr(source_text))
            except tk.TclError:
                pass
        for column, source_heading in nasa_tree_headings.items():
            nasa_tree.heading(column, text=tr(source_heading))
        for widget, source_values in widget_value_sources.items():
            if widget is language_select:
                continue
            try:
                widget.configure(values=tuple(tr(value) for value in source_values))
            except tk.TclError:
                pass
        for variable_name, source_value in variable_sources.items():
            translated_value = tr(source_value)
            if translated_value != source_value:
                root.globalsetvar(variable_name, translated_value)
        mode_var.set(tr(selected_mode))
        mode_select.configure(
            values=tuple(
                tr(name)
                for name in (
                    "Formula dinamis awal",
                    "Persamaan v2.5 literal",
                    "Referensi manual 22 N",
                )
            )
        )
        language_select.configure(
            values=tuple(language_options.values())
        )
        language_var.set(language_options[selected_code])
        title_suffix = {
            "id": "Rumus Kecepatan Fluida",
            "en": "Fluid Velocity Formula",
            "es": "Fórmula de velocidad de fluidos",
        }[selected_code]
        root.title(f"Astrolab | {title_suffix}")
        update_result()
        if current_page["name"] == "Fire Safety":
            select_fire_case()
        if current_page["name"] == "Roket vakum" and simulation_data["rocket"] is not None:
            refresh_rocket()

    def capture_ui_sources(widget) -> None:
        try:
            source_text = widget.cget("text")
            if source_text:
                widget_text_sources[widget] = source_text
        except tk.TclError:
            pass
        try:
            source_values = widget.cget("values")
            if source_values:
                widget_value_sources[widget] = tuple(source_values)
        except tk.TclError:
            pass
        try:
            variable_name = widget.cget("textvariable")
            variable_name = str(variable_name) if variable_name else ""
            if variable_name and variable_name not in variable_sources:
                variable_sources[variable_name] = str(root.globalgetvar(variable_name))
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            capture_ui_sources(child)

    widget_text_sources: dict[object, str] = {}
    widget_value_sources: dict[object, tuple[str, ...]] = {}
    variable_sources: dict[str, str] = {}
    capture_ui_sources(root)
    language_select.bind("<<ComboboxSelected>>", translate_ui)
    mode_select.bind("<<ComboboxSelected>>", update_result)
    rocket_view_select.bind(
        "<<ComboboxSelected>>",
        lambda _event: refresh_rocket() if simulation_data["rocket"] is not None else None,
    )

    def adapt_layout(event=None) -> None:
        if event is not None and event.widget is not root:
            return
        compact = root.winfo_width() < 940
        if compact == layout_state["compact"]:
            return
        layout_state["compact"] = compact
        shell.configure(padx=12 if compact else 26, pady=12 if compact else 20)
        content.columnconfigure(0, weight=1 if compact else 5, uniform="content")
        content.columnconfigure(1, weight=0 if compact else 4, uniform="content")
        if compact:
            parameter_card.grid_configure(row=0, column=0, padx=0, pady=(0, 10))
            result_card.grid_configure(row=1, column=0, padx=0)
            graph_card.grid_configure(row=1, column=0, padx=0)
            content.rowconfigure(0, weight=0)
            content.rowconfigure(1, weight=0)
            home_scroll_canvas.configure(width=max(root.winfo_width() - 34, 400))
        else:
            parameter_card.grid_configure(row=0, column=0, padx=(0, 8), pady=0)
            result_card.grid_configure(row=0, column=1, padx=(8, 0), pady=0)
            graph_card.grid_configure(row=1, column=0, columnspan=2, padx=0)
        if compact:
            graph_card.grid_configure(row=2, column=0, columnspan=1)
        else:
            graph_card.grid_configure(row=1, column=0, columnspan=2)

    layout_state = {"compact": False}
    root.bind("<Configure>", adapt_layout, add="+")

    disclaimer = label(
        shell,
        "Simulasi roket hanya ilustrasi edukatif. Bukan model aerospace tervalidasi "
        "atau untuk keputusan desain dan kendali.",
        size=9,
        color=colors["muted"],
    )
    disclaimer.grid(row=5, column=0, sticky="w", pady=(10, 0))
    show_page("Data NASA")
    def close_window() -> None:
        stop_animation("rocket")
        for callback_id in root.tk.splitlist(root.tk.call("after", "info")):
            root.after_cancel(callback_id)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_window)
    root.update_idletasks()
    root.deiconify()
    root.after(100, update_result)
    root.mainloop()


if __name__ == "__main__":
    _run_static_verification()
    import sys

    if "--cli" in sys.argv[1:]:
        run_cli()
    else:
        run_gui()
