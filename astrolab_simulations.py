"""Small educational trajectory models for Astrolab's 3D visualizer."""

from __future__ import annotations

from math import isfinite, pi

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

EARTH_RADIUS_KM = 6371.0
EARTH_MU_KM3_S2 = 398600.4418


def simulate_earth_orbit(
    altitude_km: float = 400.0,
    speed_ratio: float = 1.0,
    inclination_deg: float = 51.6,
    revolutions: float = 1.0,
    samples: int = 480,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Propagate a point mass around Earth with a two-body DOP853 solver.

    Returns elapsed seconds and an ``(samples, 3)`` array of positions in km.
    No launch, thrust, atmosphere, Earth rotation, or perturbations are modeled.
    """
    values = (altitude_km, speed_ratio, inclination_deg, revolutions)
    if not all(isfinite(value) for value in values):
        raise ValueError("Orbit parameters must be finite")
    if not 100 <= altitude_km <= 2000:
        raise ValueError("Orbit altitude must be between 100 and 2,000 km")
    if not 1.0 <= speed_ratio <= 1.05:
        raise ValueError("Circular-speed ratio must be between 1.0 and 1.05")
    if not 0 <= inclination_deg <= 180:
        raise ValueError("Inclination must be between 0 and 180 degrees")
    if not 0.25 <= revolutions <= 3:
        raise ValueError("Orbit duration must be between 0.25 and 3 revolutions")
    if not isinstance(samples, int) or isinstance(samples, bool) or not 100 <= samples <= 1500:
        raise ValueError("samples must be an integer from 100 to 1,500")

    radius = EARTH_RADIUS_KM + altitude_km
    circular_speed = (EARTH_MU_KM3_S2 / radius) ** 0.5
    inclination = inclination_deg * pi / 180
    initial_state = np.array(
        [
            radius,
            0.0,
            0.0,
            0.0,
            circular_speed * speed_ratio * np.cos(inclination),
            circular_speed * speed_ratio * np.sin(inclination),
        ],
        dtype=np.float64,
    )
    period = 2 * pi * (radius**3 / EARTH_MU_KM3_S2) ** 0.5
    times = np.linspace(0.0, period * revolutions, samples, dtype=np.float64)

    def equations(_time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        position = state[:3]
        distance = np.linalg.norm(position)
        if distance == 0:
            raise RuntimeError("Orbit propagation reached Earth's center")
        acceleration = -EARTH_MU_KM3_S2 * position / distance**3
        return np.concatenate((state[3:], acceleration))

    result = solve_ivp(
        equations,
        (float(times[0]), float(times[-1])),
        initial_state,
        method="DOP853",
        t_eval=times,
        rtol=1e-9,
        atol=1e-9,
        max_step=period / 80,
    )
    if not result.success or result.y.shape[1] != samples:
        raise RuntimeError(f"Orbit propagation failed: {result.message}")
    return result.t, result.y[:3].T.copy()
