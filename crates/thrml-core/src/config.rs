//! Path configuration for THRML.
//!
//! This module provides configurable paths for:
//! - **Cache directory**: Intermediate files, compiled shaders, etc.
//! - **Data directory**: Training data, datasets
//! - **Output directory**: Model checkpoints, visualizations, logs
//!
//! Paths can be configured via:
//! 1. CLI arguments (highest priority)
//! 2. Environment variables
//! 3. Config file (`~/.config/thrml/config.toml`)
//! 4. Default system directories
//!
//! # Example
//!
//! ```ignore
//! use thrml_core::config::PathConfig;
//!
//! // Parse from CLI args
//! let config = PathConfig::from_args();
//!
//! // Or with custom paths
//! let config = PathConfig::builder()
//!     .cache_dir("/Volumes/ExternalDisk/thrml/cache")
//!     .output_dir("/Volumes/ExternalDisk/thrml/output")
//!     .build();
//!
//! // Use paths
//! let checkpoint_path = config.output_dir().join("model_checkpoint.bin");
//! ```

use clap::Parser;
use directories::ProjectDirs;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Global path configuration instance
static GLOBAL_CONFIG: OnceCell<PathConfig> = OnceCell::new();

/// CLI arguments for path configuration
#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "THRML Path Configuration")]
pub struct PathArgs {
    /// Cache directory for intermediate files (shaders, temp data)
    #[arg(long, env = "THRML_CACHE_DIR")]
    pub cache_dir: Option<PathBuf>,

    /// Data directory for training data and datasets
    #[arg(long, env = "THRML_DATA_DIR")]
    pub data_dir: Option<PathBuf>,

    /// Output directory for checkpoints, visualizations, and logs
    #[arg(long, env = "THRML_OUTPUT_DIR")]
    pub output_dir: Option<PathBuf>,

    /// Base directory for all THRML files (overrides individual paths)
    #[arg(long, env = "THRML_BASE_DIR")]
    pub base_dir: Option<PathBuf>,

    /// Path to config file
    #[arg(long, env = "THRML_CONFIG_FILE")]
    pub config_file: Option<PathBuf>,
}

/// Path configuration from config file
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PathConfigFile {
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
    /// Data directory  
    pub data_dir: Option<PathBuf>,
    /// Output directory
    pub output_dir: Option<PathBuf>,
    /// Base directory (overrides individual paths if set)
    pub base_dir: Option<PathBuf>,
}

// =============================================================================
// Runtime Policy Configuration
// =============================================================================

/// Runtime policy configuration from config file.
///
/// Allows overriding auto-detected hardware settings via TOML config.
///
/// # Example TOML
///
/// ```toml
/// [runtime]
/// profile = "gpu-mixed"
/// real_dtype = "f32"
/// use_gpu = true
/// max_rel_error = 1e-4
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuntimePolicyConfig {
    /// Override detected profile: "auto", "cpu-fp64-strict", "gpu-mixed", "gpu-hpc-fp64"
    pub profile: Option<String>,
    /// Override real dtype: "f32" or "f64"
    pub real_dtype: Option<String>,
    /// Override complex dtype: "f32" (complex64) or "f64" (complex128)
    pub complex_dtype: Option<String>,
    /// Force GPU usage: true/false (None = auto-detect)
    pub use_gpu: Option<bool>,
    /// Maximum acceptable relative error for validation
    pub max_rel_error: Option<f64>,
}

/// Hardware-specific overrides in config file.
///
/// Allows configuring different policies for specific GPU models.
///
/// # Example TOML
///
/// ```toml
/// [hardware."Apple M3 Pro"]
/// profile = "cpu-fp64-strict"
/// max_rel_error = 1e-6
///
/// [hardware."NVIDIA GeForce RTX 5090"]
/// profile = "gpu-mixed"
/// real_dtype = "f32"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareOverrides {
    /// Overrides keyed by hardware name pattern (e.g., "Apple M*", "NVIDIA GeForce RTX 5090")
    #[serde(flatten)]
    pub overrides: std::collections::HashMap<String, RuntimePolicyConfig>,
}

/// Complete path configuration for THRML
#[derive(Debug, Clone)]
pub struct PathConfig {
    cache_dir: PathBuf,
    data_dir: PathBuf,
    output_dir: PathBuf,
}

impl PathConfig {
    /// Parse configuration from CLI arguments
    ///
    /// Priority order:
    /// 1. CLI arguments
    /// 2. Environment variables
    /// 3. Config file
    /// 4. Default directories
    pub fn from_args() -> Self {
        let args = PathArgs::parse();
        Self::from_path_args(args)
    }

    /// Parse configuration from CLI arguments (ignoring unknown args)
    ///
    /// Use this when mixing with other CLI parsers
    pub fn from_args_relaxed() -> Self {
        let args = PathArgs::try_parse().unwrap_or(PathArgs {
            cache_dir: None,
            data_dir: None,
            output_dir: None,
            base_dir: None,
            config_file: None,
        });
        Self::from_path_args(args)
    }

    /// Create configuration from PathArgs
    ///
    /// Use this when you have a flattened PathArgs in your own CLI parser.
    pub fn from_path_args(args: PathArgs) -> Self {
        // Load config file if specified or look for default
        let file_config = Self::load_config_file(args.config_file.as_deref());

        // Determine base directory
        let base_dir = args
            .base_dir
            .or(file_config.base_dir.clone())
            .or_else(|| env::var("THRML_BASE_DIR").ok().map(PathBuf::from));

        // Build paths with priority: CLI > env > file > defaults
        let defaults = Self::default_dirs();

        let cache_dir = args
            .cache_dir
            .or_else(|| base_dir.as_ref().map(|b| b.join("cache")))
            .or(file_config.cache_dir)
            .unwrap_or(defaults.0);

        let data_dir = args
            .data_dir
            .or_else(|| base_dir.as_ref().map(|b| b.join("data")))
            .or(file_config.data_dir)
            .unwrap_or(defaults.1);

        let output_dir = args
            .output_dir
            .or_else(|| base_dir.as_ref().map(|b| b.join("output")))
            .or(file_config.output_dir)
            .unwrap_or(defaults.2);

        PathConfig {
            cache_dir,
            data_dir,
            output_dir,
        }
    }

    /// Create a new builder for custom configuration
    pub fn builder() -> PathConfigBuilder {
        PathConfigBuilder::new()
    }

    /// Get the cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get the data directory
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Get the output directory
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Ensure all directories exist, creating them if necessary
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.cache_dir)?;
        fs::create_dir_all(&self.data_dir)?;
        fs::create_dir_all(&self.output_dir)?;
        Ok(())
    }

    /// Get the global configuration instance
    ///
    /// Initializes with defaults on first call. Use `set_global` to customize.
    pub fn global() -> &'static PathConfig {
        GLOBAL_CONFIG.get_or_init(PathConfig::from_args_relaxed)
    }

    /// Set the global configuration
    ///
    /// Returns Err if already initialized
    pub fn set_global(config: PathConfig) -> Result<(), PathConfig> {
        GLOBAL_CONFIG.set(config)
    }

    /// Print configuration summary
    pub fn print_summary(&self) {
        println!("THRML Path Configuration:");
        println!("  Cache:  {:?}", self.cache_dir);
        println!("  Data:   {:?}", self.data_dir);
        println!("  Output: {:?}", self.output_dir);
    }

    /// Get default directories based on OS conventions
    fn default_dirs() -> (PathBuf, PathBuf, PathBuf) {
        if let Some(proj_dirs) = ProjectDirs::from("", "", "thrml") {
            (
                proj_dirs.cache_dir().to_path_buf(),
                proj_dirs.data_dir().to_path_buf(),
                proj_dirs.data_dir().join("output"),
            )
        } else {
            // Fallback to current directory
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            (
                cwd.join(".thrml/cache"),
                cwd.join(".thrml/data"),
                cwd.join("output"),
            )
        }
    }

    /// Load config file from path or default location
    fn load_config_file(path: Option<&Path>) -> PathConfigFile {
        let config_path = path.map(PathBuf::from).or_else(|| {
            ProjectDirs::from("", "", "thrml").map(|dirs| dirs.config_dir().join("config.toml"))
        });

        if let Some(path) = config_path {
            if path.exists() {
                if let Ok(contents) = fs::read_to_string(&path) {
                    if let Ok(config) = toml::from_str::<PathConfigFile>(&contents) {
                        return config;
                    }
                }
            }
        }

        PathConfigFile::default()
    }

    /// Save current configuration to a file
    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        let config = PathConfigFile {
            cache_dir: Some(self.cache_dir.clone()),
            data_dir: Some(self.data_dir.clone()),
            output_dir: Some(self.output_dir.clone()),
            base_dir: None,
        };

        let toml_str = toml::to_string_pretty(&config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(path, toml_str)
    }

    /// Save to default config location
    pub fn save_to_default(&self) -> std::io::Result<()> {
        if let Some(proj_dirs) = ProjectDirs::from("", "", "thrml") {
            let config_path = proj_dirs.config_dir().join("config.toml");
            self.save_to_file(&config_path)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine config directory",
            ))
        }
    }
}

impl Default for PathConfig {
    fn default() -> Self {
        let (cache, data, output) = Self::default_dirs();
        PathConfig {
            cache_dir: cache,
            data_dir: data,
            output_dir: output,
        }
    }
}

/// Builder for PathConfig
#[derive(Debug, Clone, Default)]
pub struct PathConfigBuilder {
    cache_dir: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    base_dir: Option<PathBuf>,
}

impl PathConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set cache directory
    pub fn cache_dir<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Set data directory
    pub fn data_dir<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.data_dir = Some(path.into());
        self
    }

    /// Set output directory
    pub fn output_dir<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.output_dir = Some(path.into());
        self
    }

    /// Set base directory (will create cache/data/output subdirectories)
    pub fn base_dir<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.base_dir = Some(path.into());
        self
    }

    /// Build the PathConfig
    pub fn build(self) -> PathConfig {
        let defaults = PathConfig::default_dirs();

        let (cache_default, data_default, output_default) = if let Some(base) = &self.base_dir {
            (base.join("cache"), base.join("data"), base.join("output"))
        } else {
            defaults
        };

        PathConfig {
            cache_dir: self.cache_dir.unwrap_or(cache_default),
            data_dir: self.data_dir.unwrap_or(data_default),
            output_dir: self.output_dir.unwrap_or(output_default),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PathConfig::default();
        assert!(!config.cache_dir().as_os_str().is_empty());
        assert!(!config.data_dir().as_os_str().is_empty());
        assert!(!config.output_dir().as_os_str().is_empty());
    }

    #[test]
    fn test_builder() {
        let config = PathConfig::builder()
            .cache_dir("/tmp/test/cache")
            .data_dir("/tmp/test/data")
            .output_dir("/tmp/test/output")
            .build();

        assert_eq!(config.cache_dir(), Path::new("/tmp/test/cache"));
        assert_eq!(config.data_dir(), Path::new("/tmp/test/data"));
        assert_eq!(config.output_dir(), Path::new("/tmp/test/output"));
    }

    #[test]
    fn test_base_dir_builder() {
        let config = PathConfig::builder()
            .base_dir("/Volumes/External/thrml")
            .build();

        assert_eq!(
            config.cache_dir(),
            Path::new("/Volumes/External/thrml/cache")
        );
        assert_eq!(config.data_dir(), Path::new("/Volumes/External/thrml/data"));
        assert_eq!(
            config.output_dir(),
            Path::new("/Volumes/External/thrml/output")
        );
    }
}
