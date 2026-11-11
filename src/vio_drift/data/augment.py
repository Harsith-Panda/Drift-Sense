"""Small training-only observation corruptions, never temporal relabelling.

Random crop means crop-and-mask on the original timeline, not a shorter
integration interval. Zero in normalized space imputes the training mean.
Targets/auxiliary features continue to describe the original full window.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class IMUAugmentation:
    gaussian_noise: bool = False
    random_crop: bool = False
    noise_fraction: float = 0.1
    crop_max_samples: int = 4
    # Verified in both flights' imu0/sensor.yaml; continuous densities.
    gyro_noise_density: float = 1.6968e-4
    accel_noise_density: float = 2e-3
    sample_rate_hz: float = 200.0

    def __post_init__(self):
        if not 0 <= self.noise_fraction <= 0.1:
            raise ValueError("noise_fraction must be in [0, 0.1]")
        if not isinstance(self.crop_max_samples, int) or not 0 <= self.crop_max_samples <= 4:
            raise ValueError("crop_max_samples must be an integer in [0, 4]")
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.gyro_noise_density, self.accel_noise_density, self.sample_rate_hz)):
            raise ValueError("Sensor noise parameters must be finite and positive")

    @property
    def enabled(self):
        return self.gaussian_noise or self.random_crop

    def apply(self, normalized_imu, imu_std, rng, *, split):
        if split != "train":
            raise ValueError("Augmentation is allowed only on train")
        out = np.array(normalized_imu, dtype=np.float32, copy=True)
        scales = np.asarray(imu_std, dtype=np.float64)
        if out.shape != (400, 6) or scales.shape != (6,) or not np.isfinite(out).all():
            raise ValueError("Expected finite (400,6) IMU and six scales")
        if not np.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Invalid normalization scales")
        if self.gaussian_noise:
            # density / sqrt(dt); clipped to 3 sigma, at most 0.3 sensor sigma.
            sensor_std = np.array([self.gyro_noise_density] * 3 +
                                  [self.accel_noise_density] * 3) * np.sqrt(self.sample_rate_hz)
            noise = np.clip(rng.standard_normal(out.shape), -3, 3)
            out += (noise * self.noise_fraction * sensor_std / scales).astype(np.float32)
        if self.random_crop and self.crop_max_samples:
            removed = int(rng.integers(1, self.crop_max_samples + 1))
            left = int(rng.integers(0, removed + 1))
            right = removed - left
            out[:left] = 0
            if right:
                out[-right:] = 0
        return out
