//! Configuration for sphere optimization.
//!
//! This module provides scale profiles and configuration settings for
//! controlling the sphere optimization process.

use std::str::FromStr;

/// Shell capacity exponent for radial distribution.
///
/// For a sphere, shell volume at radius r is dV = 4πr² dr.
/// To achieve uniform point density per unit surface area,
/// capacity must scale as r² (surface area law).
///
/// The radius formula is: `r = min + span * normalized^(1/CAPACITY_EXPONENT)`
///
/// - `CAPACITY_EXPONENT = 2.0`: Surface area law (uniform density) - **physically correct**
/// - `CAPACITY_EXPONENT = 3.0`: Volume law (denser outer shells)
/// - `CAPACITY_EXPONENT = 1.5`: Legacy empirical value (not recommended)
pub const SHELL_CAPACITY_EXPONENT: f32 = 2.0;

/// Scale profile for different corpus sizes.
///
/// Each profile is tuned for a specific scale of data:
/// - `Dev`: Small datasets for development/testing (~1K points)
/// - `Medium`: Medium datasets (~10K points)
/// - `Large`: Large datasets (~100K points)
/// - `Planetary`: Very large datasets (~1M+ points)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScaleProfile {
    /// Development profile for small datasets (~1K points)
    #[default]
    Dev,
    /// Medium scale profile (~10K points)
    Medium,
    /// Large scale profile (~100K points)
    Large,
    /// Planetary scale for very large datasets (~1M+ points)
    Planetary,
}

impl FromStr for ScaleProfile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "dev" | "development" => Ok(Self::Dev),
            "medium" | "med" => Ok(Self::Medium),
            "large" | "lg" => Ok(Self::Large),
            "planetary" | "planet" | "huge" => Ok(Self::Planetary),
            _ => Err(format!("unknown scale profile: {s}")),
        }
    }
}

/// Configuration settings for sphere optimization.
///
/// Controls radius bounds, interaction parameters, Langevin dynamics settings,
/// and optional entropy weighting.
#[derive(Clone, Copy, Debug)]
pub struct SphereConfig {
    /// Minimum radius (for highest prominence points)
    pub min_radius: f32,
    /// Maximum radius (for lowest prominence points)
    pub max_radius: f32,
    /// Gaussian interaction radius for lateral forces
    pub interaction_radius: f32,
    /// Langevin dynamics step size (dt)
    pub step_size: f32,
    /// Temperature for Langevin noise
    pub temperature: f32,
    /// Number of Langevin steps to run
    pub n_steps: usize,
    /// Whether to weight radii by entropy (high entropy = larger radius)
    pub entropy_weighted: bool,
}

impl Default for SphereConfig {
    fn default() -> Self {
        ScaleProfile::Dev.into()
    }
}

impl From<ScaleProfile> for SphereConfig {
    fn from(profile: ScaleProfile) -> Self {
        match profile {
            ScaleProfile::Dev => Self {
                min_radius: 32.0,
                max_radius: 512.0,
                interaction_radius: 1.0,
                step_size: 0.5,
                temperature: 0.1,
                n_steps: 120,
                entropy_weighted: false,
            },
            ScaleProfile::Medium => Self {
                min_radius: 64.0,
                max_radius: 2048.0,
                interaction_radius: 1.25,
                step_size: 0.4,
                temperature: 0.08,
                n_steps: 240,
                entropy_weighted: false,
            },
            ScaleProfile::Large => Self {
                min_radius: 96.0,
                max_radius: 8192.0,
                interaction_radius: 1.5,
                step_size: 0.35,
                temperature: 0.06,
                n_steps: 480,
                entropy_weighted: false,
            },
            ScaleProfile::Planetary => Self {
                min_radius: 128.0,
                max_radius: 32768.0,
                interaction_radius: 2.0,
                step_size: 0.3,
                temperature: 0.05,
                n_steps: 960,
                entropy_weighted: false,
            },
        }
    }
}

impl SphereConfig {
    /// Enable or disable entropy-weighted radii calculation.
    ///
    /// When enabled, points with higher entropy get larger radii,
    /// pushing uncertain embeddings to the outer shell.
    pub const fn with_entropy_weighted(mut self, enabled: bool) -> Self {
        self.entropy_weighted = enabled;
        self
    }

    /// Set the number of Langevin steps.
    pub const fn with_steps(mut self, n_steps: usize) -> Self {
        self.n_steps = n_steps;
        self
    }

    /// Set the temperature for Langevin dynamics.
    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the step size (dt) for Langevin dynamics.
    pub const fn with_step_size(mut self, step_size: f32) -> Self {
        self.step_size = step_size;
        self
    }

    /// Set the interaction radius for lateral forces.
    pub const fn with_interaction_radius(mut self, radius: f32) -> Self {
        self.interaction_radius = radius;
        self
    }

    /// Set the radius bounds.
    pub const fn with_radii(mut self, min_radius: f32, max_radius: f32) -> Self {
        self.min_radius = min_radius;
        self.max_radius = max_radius;
        self
    }

    /// Calculate the radius span (max - min).
    pub fn radius_span(&self) -> f32 {
        (self.max_radius - self.min_radius).max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_profile_from_str() {
        assert_eq!("dev".parse::<ScaleProfile>().unwrap(), ScaleProfile::Dev);
        assert_eq!("MEDIUM".parse::<ScaleProfile>().unwrap(), ScaleProfile::Medium);
        assert_eq!("large".parse::<ScaleProfile>().unwrap(), ScaleProfile::Large);
        assert_eq!("planetary".parse::<ScaleProfile>().unwrap(), ScaleProfile::Planetary);
        assert!("unknown".parse::<ScaleProfile>().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = SphereConfig::from(ScaleProfile::Dev)
            .with_entropy_weighted(true)
            .with_steps(200)
            .with_temperature(0.05);

        assert!(config.entropy_weighted);
        assert_eq!(config.n_steps, 200);
        assert_eq!(config.temperature, 0.05);
    }
}
