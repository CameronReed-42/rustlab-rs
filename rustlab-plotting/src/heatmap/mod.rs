//! Heatmap visualization for matrix data
//!
//! Provides functionality for creating heatmaps from 2D arrays,
//! commonly used for correlation matrices, density plots, and
//! general matrix visualization.
//!
//! # Examples
//!
//! ```rust
//! use rustlab_plotting::heatmap::HeatmapBuilder;
//! use rustlab_plotting::colormap::ColorMap;
//! use rustlab_math::ArrayF64;
//!
//! // Create a correlation matrix heatmap
//! let matrix = ArrayF64::random((10, 10));
//! 
//! HeatmapBuilder::new(&matrix)
//!     .colormap(ColorMap::Coolwarm)
//!     .title("Correlation Matrix")
//!     .colorbar(true)
//!     .show()?;
//! ```

use crate::{Result, PlottingError};
use crate::plot::types::{Plot, PlotConfig, HeatmapData};
use crate::colormap::ColorMap;
use rustlab_math::ArrayF64;

/// Builder for creating heatmap visualizations
pub struct HeatmapBuilder<'a> {
    data: &'a ArrayF64,
    colormap: ColorMap,
    title: Option<String>,
    xlabel: Option<String>,
    ylabel: Option<String>,
    colorbar: bool,
    colorbar_label: Option<String>,
    show_values: bool,
    grid: bool,
    x_labels: Option<Vec<String>>,
    y_labels: Option<Vec<String>>,
    vmin: Option<f64>,
    vmax: Option<f64>,
    aspect_ratio: f64,
}

impl<'a> HeatmapBuilder<'a> {
    /// Create a new heatmap builder from a 2D array
    pub fn new(data: &'a ArrayF64) -> Self {
        Self {
            data,
            colormap: ColorMap::Viridis,
            title: None,
            xlabel: None,
            ylabel: None,
            colorbar: false,
            colorbar_label: None,
            show_values: false,
            grid: true,
            x_labels: None,
            y_labels: None,
            vmin: None,
            vmax: None,
            aspect_ratio: 1.0,
        }
    }
    
    /// Set the colormap
    pub fn colormap(mut self, colormap: ColorMap) -> Self {
        self.colormap = colormap;
        self
    }
    
    /// Set the plot title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
    
    /// Set the x-axis label
    pub fn xlabel(mut self, label: impl Into<String>) -> Self {
        self.xlabel = Some(label.into());
        self
    }
    
    /// Set the y-axis label
    pub fn ylabel(mut self, label: impl Into<String>) -> Self {
        self.ylabel = Some(label.into());
        self
    }
    
    /// Enable/disable colorbar
    pub fn colorbar(mut self, show: bool) -> Self {
        self.colorbar = show;
        self
    }
    
    /// Set colorbar label
    pub fn colorbar_label(mut self, label: impl Into<String>) -> Self {
        self.colorbar_label = Some(label.into());
        self.colorbar = true; // Auto-enable colorbar
        self
    }
    
    /// Show values in cells
    pub fn show_values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }
    
    /// Enable/disable grid lines
    pub fn grid(mut self, show: bool) -> Self {
        self.grid = show;
        self
    }
    
    /// Set custom x-axis tick labels
    pub fn x_labels(mut self, labels: Vec<String>) -> Self {
        self.x_labels = Some(labels);
        self
    }
    
    /// Set custom y-axis tick labels
    pub fn y_labels(mut self, labels: Vec<String>) -> Self {
        self.y_labels = Some(labels);
        self
    }
    
    /// Set minimum value for color scale
    pub fn vmin(mut self, vmin: f64) -> Self {
        self.vmin = Some(vmin);
        self
    }
    
    /// Set maximum value for color scale
    pub fn vmax(mut self, vmax: f64) -> Self {
        self.vmax = Some(vmax);
        self
    }
    
    /// Set aspect ratio (width/height of each cell)
    pub fn aspect_ratio(mut self, ratio: f64) -> Self {
        self.aspect_ratio = ratio;
        self
    }
    
    /// Build the plot
    pub fn build(self) -> Result<Plot> {
        let (rows, cols) = (self.data.nrows(), self.data.ncols());
        
        // Validate matrix has valid dimensions
        if rows == 0 || cols == 0 {
            return Err(PlottingError::InvalidData(
                "Heatmap matrix must have non-zero dimensions".to_string()
            ));
        }
        
        // Create the plot configuration
        let mut config = PlotConfig::default();
        config.title = self.title;
        config.xlabel = self.xlabel;
        config.ylabel = self.ylabel;
        config.grid = self.grid;
        
        // Calculate appropriate plot size based on matrix dimensions
        let cell_size = if rows.max(cols) > 50 { 10 } else { 20 }; // Smaller cells for large matrices
        let width = (cols * cell_size + 200).min(1200) as u32; // Add margin for colorbar, cap at 1200
        let height = (rows * cell_size + 150).min(900) as u32;  // Add margin, cap at 900
        config.width = width;
        config.height = height;
        
        // Create heatmap-specific data
        let heatmap_data = HeatmapData {
            matrix: self.data.clone(),
            colormap: self.colormap,
            vmin: self.vmin,
            vmax: self.vmax,
            show_colorbar: self.colorbar,
            colorbar_label: self.colorbar_label,
            show_values: self.show_values,
            x_labels: self.x_labels,
            y_labels: self.y_labels,
        };
        
        // Create the plot with heatmap data
        let plot = Plot {
            config,
            series: Vec::new(),
            subplots: Vec::new(),
            heatmap_data: Some(heatmap_data),
            contour_data: None,
            surface3d_data: None,
        };
        
        Ok(plot)
    }
    
    /// Show the heatmap
    pub fn show(self) -> Result<()> {
        let plot = self.build()?;
        plot.show()
    }
    
    /// Save the heatmap to a file
    pub fn save(self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let plot = self.build()?;
        plot.save(path)
    }
}

/// Convenience function for creating a simple heatmap
pub fn heatmap(data: &ArrayF64) -> Result<()> {
    HeatmapBuilder::new(data).show()
}

/// Create a correlation matrix heatmap with appropriate defaults
pub fn correlation_heatmap(data: &ArrayF64, labels: Option<Vec<String>>) -> Result<()> {
    let mut builder = HeatmapBuilder::new(data)
        .colormap(ColorMap::Coolwarm)
        .vmin(-1.0)
        .vmax(1.0)
        .colorbar(true)
        .colorbar_label("Correlation")
        .title("Correlation Matrix");
    
    if let Some(labels) = labels {
        builder = builder.x_labels(labels.clone()).y_labels(labels);
    }
    
    builder.show()
}