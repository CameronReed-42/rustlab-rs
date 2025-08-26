//! 3D Surface plotting module
//! 
//! Provides comprehensive 3D visualization capabilities including surface plots,
//! wireframes, and 3D scatter plots, with math-first ergonomics using rustlab-math types.

use crate::error::{PlottingError, Result};
use crate::plot::types::{Plot, PlotConfig};
use crate::colormap::ColorMap;
use rustlab_math::{ArrayF64, VectorF64};

/// 3D Surface plot data structure
#[derive(Debug, Clone)]
pub struct Surface3DData {
    /// X coordinate grid (can be 1D vector or 2D array)
    pub x_grid: ArrayF64,
    
    /// Y coordinate grid (can be 1D vector or 2D array)
    pub y_grid: ArrayF64,
    
    /// Z values as 2D array (height values)
    pub z_data: ArrayF64,
    
    /// Color mapping for the surface
    pub colormap: ColorMap,
    
    /// Whether to show wireframe
    pub wireframe: bool,
    
    /// Whether to show surface fill
    pub surface: bool,
    
    /// Elevation angle in degrees (default: 30)
    pub elevation: f64,
    
    /// Azimuth angle in degrees (default: 45)
    pub azimuth: f64,
    
    /// Z-axis scale factor for aspect ratio
    pub z_scale: f64,
    
    /// Z-axis label for 3D plots
    pub zlabel: Option<String>,
}

/// Builder for creating 3D surface plots with math-first API
pub struct Surface3DBuilder {
    x_grid: ArrayF64,
    y_grid: ArrayF64,
    z_data: ArrayF64,
    colormap: ColorMap,
    wireframe: bool,
    surface: bool,
    elevation: f64,
    azimuth: f64,
    z_scale: f64,
    title: Option<String>,
    xlabel: Option<String>,
    ylabel: Option<String>,
    zlabel: Option<String>,
}

impl Surface3DBuilder {
    /// Create a new 3D surface plot from 2D arrays
    /// 
    /// # Arguments
    /// * `x_grid` - X coordinates as 2D array (meshgrid format) or 1D vector
    /// * `y_grid` - Y coordinates as 2D array (meshgrid format) or 1D vector  
    /// * `z_data` - Z values as 2D array matching grid dimensions
    pub fn new(x_grid: &ArrayF64, y_grid: &ArrayF64, z_data: &ArrayF64) -> Result<Self> {
        // Validate dimensions
        let z_rows = z_data.nrows();
        let z_cols = z_data.ncols();
        
        // Handle both 1D vectors (will be expanded to meshgrid) and 2D arrays
        let x_grid = if x_grid.nrows() == 1 || x_grid.ncols() == 1 {
            // 1D vector case - expand to full grid
            // Get data as vector first
            let x_data: Vec<f64> = if x_grid.nrows() == 1 {
                // Row vector
                (0..x_grid.ncols()).map(|j| x_grid.get(0, j).unwrap()).collect()
            } else {
                // Column vector
                (0..x_grid.nrows()).map(|i| x_grid.get(i, 0).unwrap()).collect()
            };
            
            if x_data.len() != z_cols {
                return Err(PlottingError::InvalidData(
                    format!("X vector length {} doesn't match Z columns {}", x_data.len(), z_cols)
                ));
            }
            
            // Create meshgrid by repeating x values for each row
            let mut x_expanded = ArrayF64::zeros(z_rows, z_cols);
            for i in 0..z_rows {
                for j in 0..z_cols {
                    let _ = x_expanded.set(i, j, x_data[j]);
                }
            }
            x_expanded
        } else {
            // Already 2D - validate dimensions
            if x_grid.nrows() != z_rows || x_grid.ncols() != z_cols {
                return Err(PlottingError::InvalidData(
                    format!("X grid shape ({}, {}) doesn't match Z shape ({}, {})", 
                            x_grid.nrows(), x_grid.ncols(), z_rows, z_cols)
                ));
            }
            x_grid.clone()
        };
        
        let y_grid = if y_grid.nrows() == 1 || y_grid.ncols() == 1 {
            // 1D vector case - expand to full grid
            // Get data as vector first
            let y_data: Vec<f64> = if y_grid.nrows() == 1 {
                // Row vector
                (0..y_grid.ncols()).map(|j| y_grid.get(0, j).unwrap()).collect()
            } else {
                // Column vector
                (0..y_grid.nrows()).map(|i| y_grid.get(i, 0).unwrap()).collect()
            };
            
            if y_data.len() != z_rows {
                return Err(PlottingError::InvalidData(
                    format!("Y vector length {} doesn't match Z rows {}", y_data.len(), z_rows)
                ));
            }
            
            // Create meshgrid by repeating y values for each column
            let mut y_expanded = ArrayF64::zeros(z_rows, z_cols);
            for i in 0..z_rows {
                for j in 0..z_cols {
                    let _ = y_expanded.set(i, j, y_data[i]);
                }
            }
            y_expanded
        } else {
            // Already 2D - validate dimensions
            if y_grid.nrows() != z_rows || y_grid.ncols() != z_cols {
                return Err(PlottingError::InvalidData(
                    format!("Y grid shape ({}, {}) doesn't match Z shape ({}, {})",
                            y_grid.nrows(), y_grid.ncols(), z_rows, z_cols)
                ));
            }
            y_grid.clone()
        };
        
        Ok(Self {
            x_grid,
            y_grid,
            z_data: z_data.clone(),
            colormap: ColorMap::Viridis,
            wireframe: true,
            surface: true,
            elevation: 30.0,
            azimuth: 45.0,
            z_scale: 1.0,
            title: None,
            xlabel: None,
            ylabel: None,
            zlabel: None,
        })
    }
    
    /// Create from 1D vectors using meshgrid expansion
    /// 
    /// # Example
    /// ```
    /// let x = linspace!(0.0, 10.0, 50);
    /// let y = linspace!(-5.0, 5.0, 50);
    /// let z = array2d_from_function(&x, &y, |x, y| x.sin() * y.cos());
    /// Surface3DBuilder::from_vectors(&x, &y, &z)?;
    /// ```
    pub fn from_vectors(x: &VectorF64, y: &VectorF64, z: &ArrayF64) -> Result<Self> {
        // Convert vectors to 1-row arrays for consistent handling
        // Extract data using get() method to avoid contiguity issues
        let x_data: Vec<f64> = (0..x.len()).map(|i| x[i]).collect();
        let y_data: Vec<f64> = (0..y.len()).map(|i| y[i]).collect();
        
        let mut x_array = ArrayF64::zeros(1, x.len());
        let mut y_array = ArrayF64::zeros(1, y.len());
        
        for i in 0..x.len() {
            let _ = x_array.set(0, i, x_data[i]);
        }
        for i in 0..y.len() {
            let _ = y_array.set(0, i, y_data[i]);
        }
        
        Self::new(&x_array, &y_array, z)
    }
    
    /// Set the colormap for the surface
    pub fn colormap(mut self, colormap: ColorMap) -> Self {
        self.colormap = colormap;
        self
    }
    
    /// Show only wireframe (no surface fill)
    pub fn wireframe_only(mut self) -> Self {
        self.wireframe = true;
        self.surface = false;
        self
    }
    
    /// Show only surface (no wireframe)
    pub fn surface_only(mut self) -> Self {
        self.wireframe = false;
        self.surface = true;
        self
    }
    
    /// Show both wireframe and surface (default)
    pub fn wireframe_and_surface(mut self) -> Self {
        self.wireframe = true;
        self.surface = true;
        self
    }
    
    /// Set the viewing elevation angle in degrees (vertical rotation)
    pub fn elevation(mut self, degrees: f64) -> Self {
        self.elevation = degrees;
        self
    }
    
    /// Set the viewing azimuth angle in degrees (horizontal rotation)
    pub fn azimuth(mut self, degrees: f64) -> Self {
        self.azimuth = degrees;
        self
    }
    
    /// Set the Z-axis scale factor for aspect ratio control
    pub fn z_scale(mut self, scale: f64) -> Self {
        self.z_scale = scale;
        self
    }
    
    /// Set the plot title
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
    
    /// Set the X-axis label
    pub fn xlabel(mut self, label: impl Into<String>) -> Self {
        self.xlabel = Some(label.into());
        self
    }
    
    /// Set the Y-axis label
    pub fn ylabel(mut self, label: impl Into<String>) -> Self {
        self.ylabel = Some(label.into());
        self
    }
    
    /// Set the Z-axis label
    pub fn zlabel(mut self, label: impl Into<String>) -> Self {
        self.zlabel = Some(label.into());
        self
    }
    
    /// Build the plot structure
    pub fn build(self) -> Result<Plot> {
        let surface_data = Surface3DData {
            x_grid: self.x_grid,
            y_grid: self.y_grid,
            z_data: self.z_data,
            colormap: self.colormap,
            wireframe: self.wireframe,
            surface: self.surface,
            elevation: self.elevation,
            azimuth: self.azimuth,
            z_scale: self.z_scale,
            zlabel: self.zlabel,
        };
        
        let mut config = PlotConfig::default();
        config.title = self.title;
        config.xlabel = self.xlabel;
        config.ylabel = self.ylabel;
        // Note: zlabel is stored in surface_data for 3D-specific handling
        
        Ok(Plot {
            series: vec![],
            config,
            subplots: vec![],
            heatmap_data: None,
            contour_data: None,
            surface3d_data: Some(surface_data),
        })
    }
    
    /// Display the plot
    pub fn show(self) -> Result<()> {
        self.build()?.show()
    }
    
    /// Save the plot to a file
    pub fn save(self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.build()?.save(path)
    }
}

/// Convenience function for creating a simple 3D surface plot
/// 
/// # Example
/// ```
/// use rustlab_math::*;
/// use rustlab_plotting::*;
/// 
/// let x = linspace!(0.0, 10.0, 50);
/// let y = linspace!(-5.0, 5.0, 50);
/// let z = array2d_from_function(&x, &y, |x, y| x.sin() * y.cos());
/// surface3d(&x, &y, &z)?;
/// ```
pub fn surface3d(x: &VectorF64, y: &VectorF64, z: &ArrayF64) -> Result<()> {
    Surface3DBuilder::from_vectors(x, y, z)?.show()
}

/// Create a 3D wireframe plot (no surface fill)
pub fn wireframe3d(x: &VectorF64, y: &VectorF64, z: &ArrayF64) -> Result<()> {
    Surface3DBuilder::from_vectors(x, y, z)?
        .wireframe_only()
        .show()
}

/// Create a 3D surface plot with custom colormap
pub fn surface3d_colormap(
    x: &VectorF64, 
    y: &VectorF64, 
    z: &ArrayF64,
    colormap: ColorMap
) -> Result<()> {
    Surface3DBuilder::from_vectors(x, y, z)?
        .colormap(colormap)
        .show()
}

/// Helper function to create meshgrid from vectors (math-first API)
/// Returns (X_grid, Y_grid) as 2D arrays
/// 
/// # Example
/// ```
/// let x = linspace!(0.0, 10.0, 50);
/// let y = linspace!(-5.0, 5.0, 30);
/// let (x_grid, y_grid) = meshgrid(&x, &y)?;
/// ```
pub fn meshgrid(x: &VectorF64, y: &VectorF64) -> Result<(ArrayF64, ArrayF64)> {
    let nx = x.len();
    let ny = y.len();
    
    let mut x_grid = ArrayF64::zeros(ny, nx);
    let mut y_grid = ArrayF64::zeros(ny, nx);
    
    for i in 0..ny {
        for j in 0..nx {
            x_grid.set(i, j, x[j]);
            y_grid.set(i, j, y[i]);
        }
    }
    
    Ok((x_grid, y_grid))
}

/// Helper function to evaluate a function over a 2D grid
/// 
/// # Example
/// ```
/// let x = linspace!(0.0, 10.0, 50);
/// let y = linspace!(-5.0, 5.0, 50);
/// let z = array2d_from_function(&x, &y, |x, y| x.sin() * y.cos());
/// ```
pub fn array2d_from_function<F>(x: &VectorF64, y: &VectorF64, f: F) -> ArrayF64
where
    F: Fn(f64, f64) -> f64,
{
    let nx = x.len();
    let ny = y.len();
    let mut z = ArrayF64::zeros(ny, nx);
    
    for i in 0..ny {
        for j in 0..nx {
            z.set(i, j, f(x[j], y[i]));
        }
    }
    
    z
}