//! Contour plot visualization for 2D scalar fields
//!
//! Provides functionality for creating contour plots from 2D functions,
//! commonly used for optimization landscapes, level sets, and scalar field
//! visualization in scientific computing.
//!
//! # Examples
//!
//! ```rust
//! use rustlab_plotting::contour::ContourBuilder;
//! use rustlab_plotting::colormap::ColorMap;
//! use rustlab_math::{VectorF64, ArrayF64, range};
//!
//! // Create a 2D function contour plot
//! let x = range!(-2.0 => 2.0, 50);
//! let y = range!(-2.0 => 2.0, 50);
//! let z = array2d_from_function(&x, &y, |x, y| x*x - y*y);
//! 
//! ContourBuilder::new(&x, &y, &z)
//!     .levels(&[-2.0, -1.0, 0.0, 1.0, 2.0])
//!     .colormap(ColorMap::Coolwarm)
//!     .title("Saddle Function: f(x,y) = x² - y²")
//!     .filled(true)
//!     .show()?;
//! ```

use crate::{Result, PlottingError};
use crate::plot::types::{Plot, PlotConfig, Color};
use crate::colormap::{ColorMap, ColorMapper};
use rustlab_math::{VectorF64, ArrayF64};
use contour::ContourBuilder as ExternalContourBuilder;

/// Builder for creating contour plot visualizations
pub struct ContourBuilder<'a> {
    x_grid: &'a VectorF64,
    y_grid: &'a VectorF64,
    z_data: &'a ArrayF64,
    levels: Option<Vec<f64>>,
    colormap: ColorMap,
    title: Option<String>,
    xlabel: Option<String>,
    ylabel: Option<String>,
    filled: bool,
    line_width: f32,
    colorbar: bool,
    colorbar_label: Option<String>,
    grid: bool,
    n_levels: usize,
    vmin: Option<f64>,
    vmax: Option<f64>,
}

/// Data structure for contour line representation
#[derive(Debug, Clone)]
pub struct ContourLine {
    /// X coordinates of the contour line
    pub x: Vec<f64>,
    
    /// Y coordinates of the contour line
    pub y: Vec<f64>,
    
    /// Level value for this contour
    pub level: f64,
    
    /// Color for this contour line
    pub color: Color,
    
    /// Whether this contour forms a closed loop
    pub closed: bool,
}

/// Complete contour plot data structure
#[derive(Debug, Clone)]
pub struct ContourData {
    /// X-axis grid points
    pub x_grid: VectorF64,
    
    /// Y-axis grid points  
    pub y_grid: VectorF64,
    
    /// Z values on the 2D grid
    pub z_data: ArrayF64,
    
    /// Contour lines computed from the data
    pub contour_lines: Vec<ContourLine>,
    
    /// Contour levels
    pub levels: Vec<f64>,
    
    /// Colormap for visualization
    pub colormap: ColorMap,
    
    /// Whether to fill areas between contours
    pub filled: bool,
    
    /// Line width for contour lines
    pub line_width: f32,
    
    /// Whether to show a colorbar
    pub show_colorbar: bool,
    
    /// Label for the colorbar
    pub colorbar_label: Option<String>,
    
    /// Minimum value for color scaling
    pub vmin: Option<f64>,
    
    /// Maximum value for color scaling
    pub vmax: Option<f64>,
}

impl<'a> ContourBuilder<'a> {
    /// Create a new contour builder from grid data
    /// 
    /// # Arguments
    /// * `x_grid` - X-axis grid points (1D vector)
    /// * `y_grid` - Y-axis grid points (1D vector)  
    /// * `z_data` - Z values on the 2D grid (2D array: rows=y, cols=x)
    /// 
    /// # Requirements
    /// * `z_data.nrows() == y_grid.len()`
    /// * `z_data.ncols() == x_grid.len()`
    /// * All grid points must be in ascending order
    /// * Z values must be finite (no NaN or infinity)
    pub fn new(x_grid: &'a VectorF64, y_grid: &'a VectorF64, z_data: &'a ArrayF64) -> Result<Self> {
        // Validate input dimensions
        if z_data.nrows() != y_grid.len() {
            return Err(PlottingError::DimensionMismatch { 
                expected: y_grid.len(),
                found: z_data.nrows(),
                context: "Z data rows must match Y grid length".to_string()
            });
        }
        
        if z_data.ncols() != x_grid.len() {
            return Err(PlottingError::DimensionMismatch {
                expected: x_grid.len(), 
                found: z_data.ncols(),
                context: "Z data columns must match X grid length".to_string()
            });
        }
        
        // Validate that grids are sorted
        if !is_sorted(x_grid) || !is_sorted(y_grid) {
            return Err(PlottingError::InvalidData(
                "Grid points must be in ascending order".to_string()
            ));
        }
        
        Ok(Self {
            x_grid,
            y_grid,
            z_data,
            levels: None,
            colormap: ColorMap::Viridis,
            title: None,
            xlabel: None,
            ylabel: None,
            filled: false,
            line_width: 1.0,
            colorbar: false,
            colorbar_label: None,
            grid: true,
            n_levels: 10,
            vmin: None,
            vmax: None,
        })
    }
    
    /// Set specific contour levels
    pub fn levels(mut self, levels: &[f64]) -> Self {
        self.levels = Some(levels.to_vec());
        self
    }
    
    /// Set number of automatically generated levels
    pub fn n_levels(mut self, n: usize) -> Self {
        self.n_levels = n;
        self
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
    
    /// Enable/disable filled contours
    pub fn filled(mut self, filled: bool) -> Self {
        self.filled = filled;
        self
    }
    
    /// Set contour line width
    pub fn line_width(mut self, width: f32) -> Self {
        self.line_width = width;
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
    
    /// Enable/disable grid lines
    pub fn grid(mut self, show: bool) -> Self {
        self.grid = show;
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
    
    /// Build the contour plot
    pub fn build(self) -> Result<Plot> {
        // Compute contour levels if not provided
        let levels = match &self.levels {
            Some(levels) => levels.clone(),
            None => self.generate_auto_levels()?,
        };
        
        // Compute contour lines using the external contour crate
        let contour_lines = self.compute_contour_lines(&levels)?;
        
        // Create plot configuration
        let mut config = PlotConfig::default();
        config.title = self.title;
        config.xlabel = self.xlabel;
        config.ylabel = self.ylabel;
        config.grid = self.grid;
        
        // Calculate appropriate plot size based on aspect ratio
        let x_range = self.x_grid.max().unwrap() - self.x_grid.min().unwrap();
        let y_range = self.y_grid.max().unwrap() - self.y_grid.min().unwrap();
        let aspect_ratio = x_range / y_range;
        
        // Set reasonable plot dimensions maintaining aspect ratio
        if aspect_ratio > 1.5 {
            config.width = 1000;
            config.height = (600.0 / aspect_ratio) as u32;
        } else if aspect_ratio < 0.67 {
            config.width = (800.0 * aspect_ratio) as u32;
            config.height = 900;
        } else {
            config.width = 800;
            config.height = 600;
        }
        
        // Create contour-specific data
        let contour_data = ContourData {
            x_grid: self.x_grid.clone(),
            y_grid: self.y_grid.clone(),
            z_data: self.z_data.clone(),
            contour_lines,
            levels,
            colormap: self.colormap,
            filled: self.filled,
            line_width: self.line_width,
            show_colorbar: self.colorbar,
            colorbar_label: self.colorbar_label,
            vmin: self.vmin,
            vmax: self.vmax,
        };
        
        // Create the plot with contour data
        let plot = Plot {
            config,
            series: Vec::new(),
            subplots: Vec::new(),
            heatmap_data: None,
            contour_data: Some(contour_data),
            surface3d_data: None,
        };
        
        Ok(plot)
    }
    
    /// Show the contour plot
    pub fn show(self) -> Result<()> {
        let plot = self.build()?;
        plot.show()
    }
    
    /// Save the contour plot to a file
    pub fn save(self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let plot = self.build()?;
        plot.save(path)
    }
    
    /// Generate automatic contour levels based on data range
    fn generate_auto_levels(&self) -> Result<Vec<f64>> {
        let z_min = self.vmin.unwrap_or_else(|| self.z_data.min().unwrap());
        let z_max = self.vmax.unwrap_or_else(|| self.z_data.max().unwrap());
        
        if !z_min.is_finite() || !z_max.is_finite() {
            return Err(PlottingError::InvalidData(
                "Z data contains non-finite values".to_string()
            ));
        }
        
        if z_min >= z_max {
            return Err(PlottingError::InvalidData(
                "Z data has no variation (min >= max)".to_string()
            ));
        }
        
        let step = (z_max - z_min) / (self.n_levels - 1) as f64;
        let levels: Vec<f64> = (0..self.n_levels)
            .map(|i| z_min + i as f64 * step)
            .collect();
        
        Ok(levels)
    }
    
    /// Compute contour lines using a simple implementation 
    /// TODO: Replace with proper contour crate integration
    fn compute_contour_lines(&self, levels: &[f64]) -> Result<Vec<ContourLine>> {
        let mut contour_lines = Vec::new();
        
        // Get level range for color mapping
        let level_min = levels.iter().copied().fold(f64::INFINITY, f64::min);
        let level_max = levels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let color_mapper = ColorMapper::new(self.colormap, level_min, level_max);
        
        // Get grid dimensions
        let rows = self.z_data.nrows();
        let cols = self.z_data.ncols();
        
        // Convert grid coordinates to slices for easier access
        let x_coords = self.x_grid.as_slice().ok_or_else(|| 
            PlottingError::InvalidData("X grid data not contiguous".to_string()))?;
        let y_coords = self.y_grid.as_slice().ok_or_else(|| 
            PlottingError::InvalidData("Y grid data not contiguous".to_string()))?;
        
        // Simple marching squares-like implementation
        for &level in levels {
            let color = color_mapper.map(level);
            
            // Find contour line segments
            let contour_segments = self.find_contour_segments(level, rows, cols, x_coords, y_coords)?;
            
            // Create separate ContourLine for each segment
            for segment in contour_segments {
                if segment.len() >= 2 {
                    contour_lines.push(ContourLine {
                        x: segment.iter().map(|p| p.0).collect(),
                        y: segment.iter().map(|p| p.1).collect(),
                        level,
                        color,
                        closed: false,
                    });
                }
            }
        }
        
        Ok(contour_lines)
    }
    
    /// Find contour segments using proper marching squares algorithm
    fn find_contour_segments(&self, level: f64, rows: usize, cols: usize, x_coords: &[f64], y_coords: &[f64]) -> Result<Vec<Vec<(f64, f64)>>> {
        let mut all_segments = Vec::new();
        
        // Scan through the grid to find contour line segments
        for i in 0..rows-1 {
            for j in 0..cols-1 {
                // Get the four corners of the current cell
                let z00 = self.z_data.get(i, j).unwrap();
                let z01 = self.z_data.get(i, j+1).unwrap();
                let z10 = self.z_data.get(i+1, j).unwrap();
                let z11 = self.z_data.get(i+1, j+1).unwrap();
                
                // Find contour line segments within this cell
                let segments = self.marching_squares_cell(
                    i, j, level, z00, z01, z10, z11, x_coords, y_coords
                );
                
                // Each cell can produce multiple line segments
                if segments.len() >= 2 {
                    // Group points into line segments (every 2 points form a line)
                    for chunk in segments.chunks(2) {
                        if chunk.len() == 2 {
                            all_segments.push(vec![chunk[0], chunk[1]]);
                        }
                    }
                }
            }
        }
        
        Ok(all_segments)
    }
    
    /// Apply marching squares algorithm to a single cell
    fn marching_squares_cell(
        &self,
        i: usize, 
        j: usize, 
        level: f64, 
        z00: f64, z01: f64, z10: f64, z11: f64,
        x_coords: &[f64], 
        y_coords: &[f64]
    ) -> Vec<(f64, f64)> {
        // Determine the case based on which corners are above/below the level
        let case = ((z00 >= level) as u8) |
                   ((z01 >= level) as u8) << 1 |
                   ((z10 >= level) as u8) << 2 |
                   ((z11 >= level) as u8) << 3;
        
        // Edge crossing points
        let mut crossings = Vec::new();
        
        // Top edge crossing (between z00 and z01)
        if (z00 < level) != (z01 < level) {
            let t = if (z01 - z00).abs() > 1e-10 {
                (level - z00) / (z01 - z00)
            } else { 0.5 };
            let x = x_coords[j] + t * (x_coords[j+1] - x_coords[j]);
            let y = y_coords[i];
            crossings.push((0, (x, y))); // 0 = top edge
        }
        
        // Right edge crossing (between z01 and z11)
        if (z01 < level) != (z11 < level) {
            let t = if (z11 - z01).abs() > 1e-10 {
                (level - z01) / (z11 - z01)
            } else { 0.5 };
            let x = x_coords[j+1];
            let y = y_coords[i] + t * (y_coords[i+1] - y_coords[i]);
            crossings.push((1, (x, y))); // 1 = right edge
        }
        
        // Bottom edge crossing (between z11 and z10)
        if (z11 < level) != (z10 < level) {
            let t = if (z10 - z11).abs() > 1e-10 {
                (level - z11) / (z10 - z11)
            } else { 0.5 };
            let x = x_coords[j+1] + t * (x_coords[j] - x_coords[j+1]);
            let y = y_coords[i+1];
            crossings.push((2, (x, y))); // 2 = bottom edge
        }
        
        // Left edge crossing (between z10 and z00)
        if (z10 < level) != (z00 < level) {
            let t = if (z00 - z10).abs() > 1e-10 {
                (level - z10) / (z00 - z10)
            } else { 0.5 };
            let x = x_coords[j];
            let y = y_coords[i+1] + t * (y_coords[i] - y_coords[i+1]);
            crossings.push((3, (x, y))); // 3 = left edge
        }
        
        // Connect crossings based on marching squares cases
        let mut points = Vec::new();
        match crossings.len() {
            2 => {
                // Simple case: line passes through exactly 2 edges
                points.push(crossings[0].1);
                points.push(crossings[1].1);
            },
            4 => {
                // Saddle case: need to determine connectivity
                // For simplicity, connect opposite pairs
                if case == 5 || case == 10 { // Saddle cases
                    // Connect pairs based on geometry
                    points.push(crossings[0].1);
                    points.push(crossings[1].1);
                    // Add gap (None) to separate line segments - for now just add both
                    points.push(crossings[2].1);
                    points.push(crossings[3].1);
                } else {
                    // Regular 4-crossing case (shouldn't happen with proper implementation)
                    points.push(crossings[0].1);
                    points.push(crossings[1].1);
                }
            },
            _ => {
                // 0 or 1 crossing - no contour line in this cell, or 3+ (error case)
            }
        }
        
        points
    }
}

/// Convenience function for creating a simple contour plot
pub fn contour(x_grid: &VectorF64, y_grid: &VectorF64, z_data: &ArrayF64) -> Result<()> {
    ContourBuilder::new(x_grid, y_grid, z_data)?.show()
}

/// Create a contour plot with specified levels
pub fn contour_levels(
    x_grid: &VectorF64, 
    y_grid: &VectorF64, 
    z_data: &ArrayF64,
    levels: &[f64]
) -> Result<()> {
    ContourBuilder::new(x_grid, y_grid, z_data)?
        .levels(levels)
        .show()
}

/// Create a filled contour plot
pub fn contourf(x_grid: &VectorF64, y_grid: &VectorF64, z_data: &ArrayF64) -> Result<()> {
    ContourBuilder::new(x_grid, y_grid, z_data)?
        .filled(true)
        .colorbar(true)
        .show()
}

/// Helper function to check if a vector is sorted in ascending order
fn is_sorted(vec: &VectorF64) -> bool {
    for i in 1..vec.len() {
        if vec.get(i-1).unwrap() >= vec.get(i).unwrap() {
            return false;
        }
    }
    true
}

/// Create a 2D array from a mathematical function
/// 
/// # Arguments
/// * `x_grid` - X-axis grid points
/// * `y_grid` - Y-axis grid points
/// * `f` - Function to evaluate at each grid point
/// 
/// # Returns
/// 2D array where result[i][j] = f(x_grid[j], y_grid[i])
pub fn array2d_from_function<F>(
    x_grid: &VectorF64,
    y_grid: &VectorF64,
    f: F
) -> ArrayF64
where
    F: Fn(f64, f64) -> f64,
{
    let rows = y_grid.len();
    let cols = x_grid.len();
    let mut data = vec![0.0; rows * cols];
    
    for i in 0..rows {
        let y = y_grid.get(i).unwrap();
        for j in 0..cols {
            let x = x_grid.get(j).unwrap();
            data[i * cols + j] = f(x, y);
        }
    }
    
    // Use the correct ArrayF64 constructor
    let mut result = ArrayF64::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            result.set(i, j, data[i * cols + j]);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::{vec64, range};
    use approx::assert_relative_eq;
    
    #[test]
    fn test_array2d_from_function() {
        let x = vec64![0.0, 1.0, 2.0];
        let y = vec64![0.0, 1.0];
        
        // Test f(x,y) = x + y
        let z = array2d_from_function(&x, &y, |x, y| x + y);
        
        assert_eq!(z.nrows(), 2);
        assert_eq!(z.ncols(), 3);
        assert_relative_eq!(z.get(0, 0).unwrap(), 0.0); // f(0,0) = 0
        assert_relative_eq!(z.get(0, 1).unwrap(), 1.0); // f(1,0) = 1
        assert_relative_eq!(z.get(1, 1).unwrap(), 2.0); // f(1,1) = 2
    }
    
    #[test]
    fn test_contour_builder_validation() {
        let x = vec64![0.0, 1.0, 2.0];
        let y = vec64![0.0, 1.0];
        
        // Correct dimensions
        let z_correct = array2d_from_function(&x, &y, |x, y| x * y);
        assert!(ContourBuilder::new(&x, &y, &z_correct).is_ok());
        
        // Wrong dimensions - should fail
        let z_wrong = ArrayF64::zeros(3, 3); // 3x3 instead of 2x3
        assert!(ContourBuilder::new(&x, &y, &z_wrong).is_err());
    }
    
    #[test]
    fn test_is_sorted() {
        let sorted = vec64![1.0, 2.0, 3.0, 4.0];
        let unsorted = vec64![1.0, 3.0, 2.0, 4.0];
        
        assert!(is_sorted(&sorted));
        assert!(!is_sorted(&unsorted));
    }
    
    #[test]
    fn test_auto_levels_generation() {
        let x = vec64![0.0, 1.0];
        let y = vec64![0.0, 1.0];
        let z = array2d_from_function(&x, &y, |x, y| x + y); // Range: [0, 2]
        
        let builder = ContourBuilder::new(&x, &y, &z).unwrap()
            .n_levels(5);
        
        let levels = builder.generate_auto_levels().unwrap();
        assert_eq!(levels.len(), 5);
        assert_relative_eq!(levels[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(levels[4], 2.0, epsilon = 1e-10);
    }
}