"""Differentiable arbitrary-incidence RDE forward model for dual spots.

Coordinates:
  * Target rotates around the laboratory z axis.
  * theta is the azimuth of the beam-axis tilt (0 or pi for the two spots).
  * gamma is the beam-axis tilt from z.
  * phi specifies the transverse offset direction of the ring centre.
  * d and ring_radius use the same length unit; only their ratio affects the
    angular part of the frequency distribution.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def _basis_numpy(gamma_rad: np.ndarray, theta_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return beam axis b and transverse unit vectors u, v for a batch."""
    sin_gamma = np.sin(gamma_rad)
    cos_gamma = np.cos(gamma_rad)
    b = np.stack(
        [sin_gamma * np.cos(theta_rad), sin_gamma * np.sin(theta_rad), cos_gamma], axis=1
    )
    u = np.stack(
        [cos_gamma * np.cos(theta_rad), cos_gamma * np.sin(theta_rad), -sin_gamma], axis=1
    )
    v = np.tile(np.array([-np.sin(theta_rad), np.cos(theta_rad), 0.0]), (len(gamma_rad), 1))
    return b, u, v


def local_rde_frequencies_numpy(
    rpm: np.ndarray,
    gamma_deg: np.ndarray,
    d: np.ndarray,
    phi_deg: np.ndarray,
    ring_radius: np.ndarray,
    theta_rad: float,
    delta_l: float = 20.0,
    angular_samples: int = 128,
) -> np.ndarray:
    """Return signed local dual-mode beat frequencies for every ring angle.

    f = Delta_l/(2*pi) * ((rho x (Omega x r)) . b) / |rho|^2
    """
    rpm = np.asarray(rpm, dtype=np.float64).reshape(-1)
    gamma = np.deg2rad(np.asarray(gamma_deg, dtype=np.float64).reshape(-1))
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    phi = np.deg2rad(np.asarray(phi_deg, dtype=np.float64).reshape(-1))
    radius = np.asarray(ring_radius, dtype=np.float64).reshape(-1)
    if not (len(rpm) == len(gamma) == len(d) == len(phi) == len(radius)):
        raise ValueError("All forward-model inputs must have the same batch length.")

    alpha = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    b, u, v = _basis_numpy(gamma, theta_rad)
    rho = radius[:, None, None] * (
        np.cos(alpha)[None, :, None] * u[:, None, :] + np.sin(alpha)[None, :, None] * v[:, None, :]
    )
    centre = np.stack([d * np.cos(phi), d * np.sin(phi), np.zeros_like(d)], axis=1)
    position = centre[:, None, :] + rho
    omega = np.zeros_like(position)
    omega[:, :, 2] = 2.0 * np.pi * rpm[:, None] / 60.0
    velocity = np.cross(omega, position)
    angular_rate = np.sum(np.cross(rho, velocity) * b[:, None, :], axis=2)
    angular_rate /= np.maximum(radius[:, None] ** 2, 1e-12)
    return delta_l * angular_rate / (2.0 * np.pi)


class DualSpotRDEForward(torch.nn.Module):
    """Torch form of the same local-frequency equation for a network loss."""

    def __init__(self, delta_l: float = 20.0, angular_samples: int = 64) -> None:
        super().__init__()
        alpha = torch.linspace(0.0, 2.0 * math.pi, angular_samples + 1)[:-1]
        self.register_buffer("alpha", alpha)
        self.delta_l = float(delta_l)

    def local_frequencies(
        self,
        rpm: torch.Tensor,
        gamma_deg: torch.Tensor,
        d: torch.Tensor,
        phi_deg: torch.Tensor,
        ring_radius: torch.Tensor,
        theta_rad: float,
    ) -> torch.Tensor:
        gamma = torch.deg2rad(gamma_deg)
        phi = torch.deg2rad(phi_deg)
        sin_gamma, cos_gamma = torch.sin(gamma), torch.cos(gamma)
        theta = torch.as_tensor(theta_rad, dtype=rpm.dtype, device=rpm.device)
        b = torch.stack([sin_gamma * torch.cos(theta), sin_gamma * torch.sin(theta), cos_gamma], dim=1)
        u = torch.stack([cos_gamma * torch.cos(theta), cos_gamma * torch.sin(theta), -sin_gamma], dim=1)
        v = torch.stack(
            [-torch.sin(theta).expand_as(rpm), torch.cos(theta).expand_as(rpm), torch.zeros_like(rpm)], dim=1
        )
        rho = ring_radius[:, None, None] * (
            torch.cos(self.alpha)[None, :, None] * u[:, None, :]
            + torch.sin(self.alpha)[None, :, None] * v[:, None, :]
        )
        centre = torch.stack([d * torch.cos(phi), d * torch.sin(phi), torch.zeros_like(d)], dim=1)
        position = centre[:, None, :] + rho
        omega = torch.stack([torch.zeros_like(rpm), torch.zeros_like(rpm), 2.0 * math.pi * rpm / 60.0], dim=1)
        velocity = torch.linalg.cross(omega[:, None, :].expand_as(position), position, dim=2)
        angular_rate = torch.sum(torch.linalg.cross(rho, velocity, dim=2) * b[:, None, :], dim=2)
        angular_rate = angular_rate / torch.clamp(ring_radius[:, None].square(), min=1e-12)
        return self.delta_l * angular_rate / (2.0 * math.pi)

    def quantiles(
        self, frequencies: torch.Tensor, quantile_levels: tuple[float, ...] = (0.05, 0.50, 0.95)
    ) -> torch.Tensor:
        # Real-valued detector PSD folds negative beat frequencies onto |f|.
        return torch.quantile(torch.abs(frequencies), torch.tensor(quantile_levels, device=frequencies.device), dim=1).T
