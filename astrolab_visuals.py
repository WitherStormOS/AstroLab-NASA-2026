"""Low-resolution, vectorized Earth rendering helpers for Astrolab."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatArray = NDArray[np.float32]


def load_earth_texture(size: tuple[int, int] = (512, 256)) -> FloatArray:
    """Load the supplied equirectangular map at a bounded render resolution."""
    bundle_root = Path(
        getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)
    )
    texture_path = bundle_root / "assets" / "earth_daymap.jpg"
    with Image.open(texture_path) as source:
        image = source.convert("RGB").resize(size, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def render_earth_orthographic(
    texture: FloatArray,
    size: int = 320,
    rotation_rad: float = 0.0,
) -> FloatArray:
    """Project the map into a shaded, transparent orthographic globe image."""
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    radius_squared = x * x + y * y
    visible = radius_squared <= 1.0
    z = np.sqrt(np.clip(1.0 - radius_squared, 0.0, 1.0))

    longitude = np.arctan2(x, z) + rotation_rad
    latitude = np.arcsin(np.clip(y, -1.0, 1.0))
    texture_height, texture_width, _ = texture.shape
    u = np.mod(longitude + np.pi, 2.0 * np.pi) / (2.0 * np.pi)
    column = np.rint(u * (texture_width - 1)).astype(np.int32)
    row = np.rint(
        (np.pi / 2.0 - latitude) / np.pi * (texture_height - 1)
    ).astype(np.int32)

    light = np.asarray((0.42, -0.35, 0.84), dtype=np.float32)
    light /= np.linalg.norm(light)
    illumination = np.clip(
        x * light[0] + y * light[1] + z * light[2], 0.0, 1.0
    )
    shade = np.float32(0.28) + np.float32(0.72) * illumination
    rgb = texture[row, column] * shade[..., None]

    rim = np.clip((np.float32(1.0) - z) / np.float32(0.24), 0.0, 1.0) ** 2
    atmosphere_blue = np.asarray((0.12, 0.56, 0.92), dtype=np.float32)
    rgb = rgb * (1.0 - 0.34 * rim[..., None]) + atmosphere_blue * (
        0.34 * rim[..., None]
    )

    image = np.zeros((size, size, 4), dtype=np.float32)
    image[..., :3] = np.where(visible[..., None], rgb, 0.0)
    image[..., 3] = visible.astype(np.float32)
    return image


def earth_sphere_mesh(
    texture: FloatArray,
    *,
    radius_km: float,
    longitude_samples: int = 49,
    latitude_samples: int = 25,
    rotation_rad: float = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Create a small shaded texture mesh and a translucent cloud shell."""
    longitude = np.linspace(
        -np.pi, np.pi, longitude_samples, dtype=np.float32
    ) + np.float32(rotation_rad)
    latitude = np.linspace(
        np.pi / 2.0, -np.pi / 2.0, latitude_samples, dtype=np.float32
    )
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    cos_latitude = np.cos(lat_grid)
    unit_x = cos_latitude * np.cos(lon_grid)
    unit_y = cos_latitude * np.sin(lon_grid)
    unit_z = np.sin(lat_grid)
    radius = np.float32(radius_km)
    x, y, z = radius * unit_x, radius * unit_y, radius * unit_z

    texture_height, texture_width, _ = texture.shape
    texture_u = np.mod(lon_grid + np.pi, 2.0 * np.pi) / (2.0 * np.pi)
    columns = np.rint(texture_u * (texture_width - 1)).astype(np.int32)
    rows = np.rint(
        (np.pi / 2.0 - lat_grid) / np.pi * (texture_height - 1)
    ).astype(np.int32)
    light = np.asarray((0.42, -0.35, 0.84), dtype=np.float32)
    light /= np.linalg.norm(light)
    illumination = np.clip(
        unit_x * light[0] + unit_y * light[1] + unit_z * light[2],
        0.0,
        1.0,
    )
    shade = np.float32(0.28) + np.float32(0.72) * illumination
    surface_rgb = texture[rows, columns] * shade[..., None]
    surface_rgba = np.concatenate(
        (surface_rgb, np.ones((*surface_rgb.shape[:2], 1), dtype=np.float32)),
        axis=2,
    )

    brightness = np.mean(texture[rows, columns], axis=2)
    cloud_alpha = np.clip((brightness - 0.78) * 0.55, 0.0, 0.24)
    clouds_rgba = np.empty((*brightness.shape, 4), dtype=np.float32)
    clouds_rgba[..., :3] = 0.88
    clouds_rgba[..., 3] = cloud_alpha

    return x, y, z, surface_rgba, clouds_rgba


def visibility_mask(
    positions: NDArray[np.float64], camera_direction: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """Classify orbit segments facing toward the current camera."""
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("positions must have shape (n, 3) with n >= 2")
    if camera_direction.shape != (3,) or not np.isfinite(camera_direction).all():
        raise ValueError("camera_direction must be a finite 3D vector")
    direction_norm = np.linalg.norm(camera_direction)
    if direction_norm == 0:
        raise ValueError("camera_direction must be non-zero")
    direction = camera_direction / direction_norm
    midpoints = (positions[:-1] + positions[1:]) * 0.5
    return (midpoints @ direction) >= 0.0
