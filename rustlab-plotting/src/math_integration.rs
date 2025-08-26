//! Math-first plotting integration
//! 
//! This module provides ergonomic plotting methods that integrate tightly with
//! rustlab-math's existing mathematical functions and utilities.

use crate::{Plot, Result};
use rustlab_math::{
    VectorF64, ArrayF64, linspace,
    AxisReductions, Axis
};

/// Extension trait to add convenience methods for plotting integration
pub trait VectorDataExt {
    /// Get underlying data as slice for plotting (convenience method)
    fn data(&self) -> &[f64];
}

impl VectorDataExt for VectorF64 {
    fn data(&self) -> &[f64] {
        self.as_slice().expect("Vector must be contiguous for plotting operations")
    }
}


/// Trait for plotting expressions directly
pub trait PlotExpression {
    /// Plot this expression by evaluating it
    fn plot(&self) -> Result<()>;
    
    /// Plot this expression with a custom plot builder
    fn plot_with<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(Plot) -> Plot;
}

/// Trait for plotting mathematical functions
pub trait PlotMath {
    /// Plot a mathematical function over a range
    fn plot_function<F>(&self, f: F, label: &str) -> Result<()>
    where
        F: Fn(f64) -> f64;
    
    /// Plot multiple mathematical functions
    fn plot_functions<F>(&self, functions: &[(F, &str)]) -> Result<()>
    where
        F: Fn(f64) -> f64;
}

/// Extension trait for VectorF64 with plotting capabilities
pub trait VectorPlotExt {
    /// Plot this vector as Y values with auto-generated X indices
    fn plot_indexed(&self) -> Result<()>;
    
    /// Plot this vector against another vector
    fn plot_against(&self, other: &VectorF64) -> Result<()>;
    
    /// Create a scatter plot
    fn scatter_against(&self, other: &VectorF64) -> Result<()>;
    
    /// Plot histogram of the data
    fn plot_histogram(&self, bins: usize) -> Result<()>;
    
    /// Plot this vector and its mathematical transformations
    fn plot_with_transforms(&self, transforms: &[(&str, fn(&VectorF64) -> VectorF64)]) -> Result<()>;
}

/// Extension trait for ArrayF64 with plotting capabilities
pub trait ArrayPlotExt {
    /// Plot columns of the array
    fn plot_columns(&self) -> Result<()>;
    
    /// Plot rows of the array
    fn plot_rows(&self) -> Result<()>;
    
    /// Create a heatmap of the array
    fn heatmap(&self) -> Result<()>;
    
    /// Plot array statistics (mean, std per column/row)
    fn plot_statistics(&self, axis: Axis) -> Result<()>;
}

impl VectorPlotExt for VectorF64 {
    fn plot_indexed(&self) -> Result<()> {
        let x_data: Vec<f64> = (0..self.len()).map(|i| i as f64).collect();
        let x = VectorF64::from_slice(&x_data);
        Plot::new().line(&x, self).show()
    }
    
    fn plot_against(&self, other: &VectorF64) -> Result<()> {
        Plot::new().line(other, self).show()
    }
    
    fn scatter_against(&self, other: &VectorF64) -> Result<()> {
        Plot::new().scatter(other, self).show()
    }
    
    fn plot_histogram(&self, bins: usize) -> Result<()> {
        Plot::new().histogram(self, bins).show()
    }
    
    fn plot_with_transforms(&self, transforms: &[(&str, fn(&VectorF64) -> VectorF64)]) -> Result<()> {
        let mut plot = Plot::new();
        let x_data: Vec<f64> = (0..self.len()).map(|i| i as f64).collect();
        let x = VectorF64::from_slice(&x_data);
        
        // Plot original
        plot = plot.line_with(&x, self, "Original");
        
        // Plot each transform
        for (label, transform) in transforms {
            let transformed = transform(self);
            plot = plot.line_with(&x, &transformed, *label);
        }
        
        plot.legend(true).show()
    }
}

impl ArrayPlotExt for ArrayF64 {
    fn plot_columns(&self) -> Result<()> {
        let mut plot = Plot::new();
        let x_data: Vec<f64> = (0..self.nrows()).map(|i| i as f64).collect();
        let x = VectorF64::from_slice(&x_data);
        
        for col in 0..self.ncols() {
            // Extract column data
            let mut col_data = Vec::with_capacity(self.nrows());
            for row in 0..self.nrows() {
                col_data.push(self.get(row, col).unwrap());
            }
            let col_vec = VectorF64::from_slice(&col_data);
            plot = plot.line_with(&x, &col_vec, &format!("Column {}", col));
        }
        
        plot.legend(true).show()
    }
    
    fn plot_rows(&self) -> Result<()> {
        let mut plot = Plot::new();
        let x_data: Vec<f64> = (0..self.ncols()).map(|i| i as f64).collect();
        let x = VectorF64::from_slice(&x_data);
        
        for row in 0..self.nrows() {
            // Extract row data
            let mut row_data = Vec::with_capacity(self.ncols());
            for col in 0..self.ncols() {
                row_data.push(self.get(row, col).unwrap());
            }
            let row_vec = VectorF64::from_slice(&row_data);
            plot = plot.line_with(&x, &row_vec, &format!("Row {}", row));
        }
        
        plot.legend(true).show()
    }
    
    fn heatmap(&self) -> Result<()> {
        // TODO: Implement heatmap visualization
        // This would require additional backend support
        unimplemented!("Heatmap visualization not yet implemented")
    }
    
    fn plot_statistics(&self, axis: Axis) -> Result<()> {
        match axis {
            Axis::Rows => {
                let means = self.mean_axis(axis).expect("Failed to compute means");
                let stds = self.std_axis(axis).expect("Failed to compute std devs");
                
                let x_data: Vec<f64> = (0..means.len()).map(|i| i as f64).collect();
                let x = VectorF64::from_slice(&x_data);
                
                Plot::new()
                    .line_with(&x, &means, "Mean")
                    .line_with(&x, &stds, "Std Dev")
                    .legend(true)
                    .ylabel("Value")
                    .xlabel("Column Index")
                    .title("Column Statistics (along rows)")
                    .show()
            },
            Axis::Cols => {
                let means = self.mean_axis(axis).expect("Failed to compute means");
                let stds = self.std_axis(axis).expect("Failed to compute std devs");
                
                let x_data: Vec<f64> = (0..means.len()).map(|i| i as f64).collect();
                let x = VectorF64::from_slice(&x_data);
                
                Plot::new()
                    .line_with(&x, &means, "Mean")
                    .line_with(&x, &stds, "Std Dev")
                    .legend(true)
                    .ylabel("Value")
                    .xlabel("Row Index")
                    .title("Row Statistics (along columns)")
                    .show()
            }
        }
    }
}

/// Convenience functions for common plotting patterns
pub mod patterns {
    use super::*;
    
    /// Plot a mathematical function over a range
    pub fn plot_function<F>(f: F, start: f64, end: f64, points: usize, label: &str) -> Result<()>
    where
        F: Fn(f64) -> f64
    {
        let x = linspace(start, end, points);
        let y_data: Vec<f64> = x.as_slice().unwrap().iter().map(|&xi| f(xi)).collect();
        let y = VectorF64::from_slice(&y_data);
        
        Plot::new()
            .line_with(&x, &y, label)
            .title(label)
            .show()
    }
    
    /// Compare multiple functions on the same plot using ergonomic builder pattern
    pub fn new_comparison(start: f64, end: f64, points: usize) -> FunctionComparison {
        FunctionComparison::new(start, end, points)
    }
    
    /// Compare multiple functions on the same plot (legacy function - use new_comparison for better ergonomics)
    pub fn compare_functions<F>(
        functions: &[(F, &str)],
        start: f64,
        end: f64,
        points: usize,
    ) -> Result<()>
    where
        F: Fn(f64) -> f64
    {
        let x = linspace(start, end, points);
        let mut plot = Plot::new();
        
        for (f, label) in functions {
            let y_data: Vec<f64> = x.as_slice().unwrap().iter().map(|&xi| f(xi)).collect();
            let y = VectorF64::from_slice(&y_data);
            plot = plot.line_with(&x, &y, *label);
        }
        
        plot.legend(true)
            .title("Function Comparison")
            .show()
    }
    
    /// Ergonomic builder for comparing multiple functions with different closure types
    pub struct FunctionComparison {
        x: VectorF64,
        plot: Plot,
    }
    
    impl FunctionComparison {
        pub fn new(start: f64, end: f64, points: usize) -> Self {
            let x = linspace(start, end, points);
            let plot = Plot::new().legend(true).title("Function Comparison");
            Self { x, plot }
        }
        
        /// Add a function to compare (each can be a different closure type)
        pub fn add_function<F>(mut self, f: F, label: &str) -> Self
        where
            F: Fn(f64) -> f64
        {
            let y_data: Vec<f64> = self.x.as_slice().unwrap().iter().map(|&xi| f(xi)).collect();
            let y = VectorF64::from_slice(&y_data);
            self.plot = self.plot.line_with(&self.x, &y, label);
            self
        }
        
        /// Show the comparison plot
        pub fn show(self) -> Result<()> {
            self.plot.show()
        }
    }
    
    /// Plot vector field (2D)
    pub fn plot_vector_field<F>(
        f: F,
        x_range: (f64, f64),
        y_range: (f64, f64),
        grid_size: usize,
    ) -> Result<()>
    where
        F: Fn(f64, f64) -> (f64, f64)
    {
        // TODO: Implement vector field visualization
        unimplemented!("Vector field visualization not yet implemented")
    }
    
    /// Plot parametric curve with ergonomic API accepting different closure types
    pub fn plot_parametric<Fx, Fy>(
        x_func: Fx,
        y_func: Fy,
        t_start: f64,
        t_end: f64,
        points: usize,
        label: &str,
    ) -> Result<()>
    where
        Fx: Fn(f64) -> f64,
        Fy: Fn(f64) -> f64,
    {
        let t = linspace(t_start, t_end, points);
        let x_data: Vec<f64> = t.as_slice().unwrap().iter().map(|&ti| x_func(ti)).collect();
        let x = VectorF64::from_slice(&x_data);
        let y_data: Vec<f64> = t.as_slice().unwrap().iter().map(|&ti| y_func(ti)).collect();
        let y = VectorF64::from_slice(&y_data);
        
        Plot::new()
            .line_with(&x, &y, label)
            .title(label)
            .show()
    }
}

/// Math-aware plot builder extensions
pub trait MathPlotBuilder {
    /// Add a mathematical function to the plot
    fn function<F>(self, f: F, x: &VectorF64, label: &str) -> Self
    where
        F: Fn(f64) -> f64;
    
    // Note: Expression system not yet implemented in rustlab-math
    // /// Add expression result to the plot
    // fn expression<E: Expression>(self, expr: &E, label: &str) -> Self;
    
    /// Add mathematical transformation of existing data
    fn transform<F>(self, x: &VectorF64, y: &VectorF64, f: F, label: &str) -> Self
    where
        F: Fn(f64) -> f64;
}

impl MathPlotBuilder for Plot {
    fn function<F>(self, f: F, x: &VectorF64, label: &str) -> Self
    where
        F: Fn(f64) -> f64
    {
        let y_data: Vec<f64> = x.as_slice().unwrap().iter().map(|&xi| f(xi)).collect();
        let y = VectorF64::from_slice(&y_data);
        self.line_with(x, &y, label)
    }
    
    // Note: Expression system not yet implemented in rustlab-math
    // fn expression<E: Expression>(self, expr: &E, label: &str) -> Self {
    //     // This would require Expression to provide evaluation methods
    //     // For now, we'll leave it unimplemented
    //     unimplemented!("Expression plotting not yet implemented")
    // }
    
    fn transform<F>(self, x: &VectorF64, y: &VectorF64, f: F, label: &str) -> Self
    where
        F: Fn(f64) -> f64
    {
        let transformed_y_data: Vec<f64> = y.as_slice().unwrap().iter().map(|&yi| f(yi)).collect();
        let transformed_y = VectorF64::from_slice(&transformed_y_data);
        self.line_with(x, &transformed_y, label)
    }
}