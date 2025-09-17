//! Color mapping functionality for scientific visualization
//! 
//! Provides perceptually uniform colormaps for data visualization,
//! particularly important for heatmaps, contour plots, and 3D surfaces.
//!
//! # Available Colormaps
//!
//! - **Viridis**: Perceptually uniform, colorblind-friendly (default)
//! - **Plasma**: Purple-pink-yellow, good for dark backgrounds
//! - **Inferno**: Black-red-yellow, high contrast
//! - **Magma**: Black-purple-white, good for presentations
//! - **Turbo**: Rainbow-like but with better perceptual properties
//! - **Coolwarm**: Blue-white-red diverging for centered data
//! - **Greys**: Simple grayscale
//!
//! # Example
//!
//! ```rust
//! use rustlab_plotting::colormap::{ColorMap, ColorMapper};
//!
//! // Create a color mapper for data range [0, 100]
//! let mapper = ColorMapper::new(ColorMap::Viridis, 0.0, 100.0);
//! 
//! // Map a value to RGB color
//! let color = mapper.map(50.0);  // Returns Color at midpoint
//! ```

use crate::plot::types::Color;

/// Available colormap schemes for scientific visualization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorMap {
    /// Perceptually uniform, colorblind-friendly (matplotlib default)
    Viridis,
    /// Purple-pink-yellow, good contrast
    Plasma,
    /// Black-red-yellow, high contrast
    Inferno,
    /// Black-purple-white, good for presentations
    Magma,
    /// Improved rainbow (Google Turbo)
    Turbo,
    /// Blue-white-red diverging colormap
    Coolwarm,
    /// Simple grayscale
    Greys,
    /// Red-yellow-green-blue (Jet/Rainbow) - use with caution
    Jet,
}

impl ColorMap {
    /// Get the RGB values for a normalized position [0, 1]
    pub fn get_color(&self, t: f64) -> Color {
        let t = t.max(0.0).min(1.0); // Clamp to [0, 1]
        
        match self {
            ColorMap::Viridis => self.viridis_color(t),
            ColorMap::Plasma => self.plasma_color(t),
            ColorMap::Inferno => self.inferno_color(t),
            ColorMap::Magma => self.magma_color(t),
            ColorMap::Turbo => self.turbo_color(t),
            ColorMap::Coolwarm => self.coolwarm_color(t),
            ColorMap::Greys => self.grey_color(t),
            ColorMap::Jet => self.jet_color(t),
        }
    }
    
    /// Viridis colormap (approximation)
    fn viridis_color(&self, t: f64) -> Color {
        let r = (0.267 + t * (0.004 + t * (1.351 - t * 1.622))) * 255.0;
        let g = (0.004 + t * (1.328 + t * (-0.330 - t * 0.002))) * 255.0;
        let b = (0.329 + t * (1.787 - t * (2.947 - t * 1.831))) * 255.0;
        
        Color {
            r: r.max(0.0).min(255.0) as u8,
            g: g.max(0.0).min(255.0) as u8,
            b: b.max(0.0).min(255.0) as u8,
        }
    }
    
    /// Plasma colormap (approximation)
    fn plasma_color(&self, t: f64) -> Color {
        let r = (0.050 + t * (2.358 - t * (2.503 - t * 1.095))) * 255.0;
        let g = (0.004 + t * (0.027 + t * (1.419 - t * 1.450))) * 255.0;
        let b = (0.533 + t * (1.363 - t * (2.368 - t * 1.472))) * 255.0;
        
        Color {
            r: r.max(0.0).min(255.0) as u8,
            g: g.max(0.0).min(255.0) as u8,
            b: b.max(0.0).min(255.0) as u8,
        }
    }
    
    /// Inferno colormap (approximation)
    fn inferno_color(&self, t: f64) -> Color {
        let r = (t * (2.025 + t * (0.756 - t * 0.781))) * 255.0;
        let g = (t * (0.014 + t * (1.671 - t * 1.685))) * 255.0;
        let b = (0.016 + t * (1.932 - t * (3.170 - t * 2.218))) * 255.0;
        
        Color {
            r: r.max(0.0).min(255.0) as u8,
            g: g.max(0.0).min(255.0) as u8,
            b: b.max(0.0).min(255.0) as u8,
        }
    }
    
    /// Magma colormap (approximation)
    fn magma_color(&self, t: f64) -> Color {
        let r = (t * (1.459 + t * (1.091 - t * 0.550))) * 255.0;
        let g = (t * (0.033 + t * (1.389 - t * 1.422))) * 255.0;
        let b = (0.014 + t * (1.744 - t * (2.479 - t * 1.749))) * 255.0;
        
        Color {
            r: r.max(0.0).min(255.0) as u8,
            g: g.max(0.0).min(255.0) as u8,
            b: b.max(0.0).min(255.0) as u8,
        }
    }
    
    /// Turbo colormap (Google's improved rainbow)
    fn turbo_color(&self, t: f64) -> Color {
        // Simplified turbo approximation
        let r = if t < 0.35 {
            0.237 - 1.142 * t + 4.914 * t * t
        } else if t < 0.67 {
            2.0 - 2.0 * t
        } else {
            1.659 * t - 0.659
        };
        
        let g = if t < 0.5 {
            1.321 * t + 0.044
        } else {
            2.064 - 1.744 * t
        };
        
        let b = if t < 0.34 {
            1.542 * t + 0.167
        } else if t < 0.65 {
            1.0
        } else {
            3.874 - 3.874 * t
        };
        
        Color {
            r: (r * 255.0).max(0.0).min(255.0) as u8,
            g: (g * 255.0).max(0.0).min(255.0) as u8,
            b: (b * 255.0).max(0.0).min(255.0) as u8,
        }
    }
    
    /// Cool-warm diverging colormap
    fn coolwarm_color(&self, t: f64) -> Color {
        let r = ((1.0 + t) / 2.0 * 255.0) as u8;
        let g = (0.5 * 255.0) as u8;
        let b = ((2.0 - t) / 2.0 * 255.0) as u8;
        
        Color { r, g, b }
    }
    
    /// Simple greyscale
    fn grey_color(&self, t: f64) -> Color {
        let v = (t * 255.0) as u8;
        Color { r: v, g: v, b: v }
    }
    
    /// Jet colormap (rainbow) - included for compatibility but not recommended
    fn jet_color(&self, t: f64) -> Color {
        let r = if t < 0.375 {
            0.0
        } else if t < 0.625 {
            4.0 * (t - 0.375)
        } else if t < 0.875 {
            1.0
        } else {
            1.0 - 4.0 * (t - 0.875)
        };
        
        let g = if t < 0.125 {
            0.0
        } else if t < 0.375 {
            4.0 * (t - 0.125)
        } else if t < 0.625 {
            1.0
        } else if t < 0.875 {
            1.0 - 4.0 * (t - 0.625)
        } else {
            0.0
        };
        
        let b = if t < 0.125 {
            0.5 + 4.0 * t
        } else if t < 0.375 {
            1.0
        } else if t < 0.625 {
            1.0 - 4.0 * (t - 0.375)
        } else {
            0.0
        };
        
        Color {
            r: (r * 255.0) as u8,
            g: (g * 255.0) as u8,
            b: (b * 255.0) as u8,
        }
    }
}

/// Maps data values to colors using a specified colormap
pub struct ColorMapper {
    colormap: ColorMap,
    min_val: f64,
    #[allow(dead_code)]
    max_val: f64,
    range: f64,
}

impl ColorMapper {
    /// Create a new color mapper for the given data range
    pub fn new(colormap: ColorMap, min_val: f64, max_val: f64) -> Self {
        let range = max_val - min_val;
        Self {
            colormap,
            min_val,
            max_val,
            range: if range.abs() < 1e-10 { 1.0 } else { range },
        }
    }
    
    /// Map a data value to a color
    pub fn map(&self, value: f64) -> Color {
        let t = (value - self.min_val) / self.range;
        self.colormap.get_color(t)
    }
    
    /// Get a discrete set of colors for contour levels
    pub fn get_colors(&self, n_levels: usize) -> Vec<Color> {
        (0..n_levels)
            .map(|i| {
                let t = i as f64 / (n_levels - 1).max(1) as f64;
                self.colormap.get_color(t)
            })
            .collect()
    }
}

/// Generate a colorbar legend for the colormap
pub fn generate_colorbar_data(colormap: ColorMap, n_samples: usize) -> Vec<Color> {
    (0..n_samples)
        .map(|i| {
            let t = i as f64 / (n_samples - 1) as f64;
            colormap.get_color(t)
        })
        .collect()
}