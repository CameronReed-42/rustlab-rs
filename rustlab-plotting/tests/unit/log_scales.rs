//! Unit tests for logarithmic scale functionality

use rustlab_plotting::prelude::*;
use rustlab_plotting::plot::validation::{validate_plot_data, filter_positive_data, adjust_log_range};
use rustlab_plotting::VectorDataExt;
use rustlab_core::*;

#[test]
fn test_linear_scale_works() {
    let x = linspace(0.0, 10.0, 11);
    let y = x.map_f64(|x| x * x);
    
    let result = Plot::new()
        .title("Linear Scale Test")
        .line(&x, &y)
        .save("test_linear.png");
    
    assert!(result.is_ok(), "Linear scale should work with any data");
}

#[test]
fn test_log10_y_scale_with_positive_data() {
    let x = linspace(1.0, 10.0, 10);
    let y = x.map_f64(|x| x.powf(2.0)); // Always positive
    
    let result = Plot::new()
        .title("Log10 Y Scale Test")
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save("test_log10_y.png");
    
    assert!(result.is_ok(), "Log10 Y scale should work with positive data");
}

#[test]
fn test_log10_x_scale_with_positive_data() {
    let x = linspace(1.0, 100.0, 50);
    let y = x.map_f64(|x| x.sqrt()); // Always positive
    
    let result = Plot::new()
        .title("Log10 X Scale Test")
        .xscale(Scale::Log10)
        .line(&x, &y)
        .save("test_log10_x.png");
    
    assert!(result.is_ok(), "Log10 X scale should work with positive data");
}

#[test]
fn test_log10_both_axes_with_positive_data() {
    let x = linspace(1.0, 100.0, 50);
    let y = x.map_f64(|x| x.powf(1.5)); // Always positive
    
    let result = Plot::new()
        .title("Log10 Both Axes Test")
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save("test_log10_both.png");
    
    assert!(result.is_ok(), "Log10 both axes should work with positive data");
}

#[test]
fn test_log_scale_rejects_negative_x_data() {
    let x = linspace(-5.0, 5.0, 11); // Contains negative values
    let y = x.map_f64(|x| x.abs());
    
    let result = Plot::new()
        .title("Log Scale with Negative X")
        .xscale(Scale::Log10)
        .line(&x, &y)
        .save("test_log_negative_x.png");
    
    assert!(result.is_err(), "Log X scale should reject negative data");
}

#[test]
fn test_log_scale_rejects_negative_y_data() {
    let x = linspace(1.0, 10.0, 10);
    let y = linspace(-5.0, 5.0, 10); // Contains negative values
    
    let result = Plot::new()
        .title("Log Scale with Negative Y")
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save("test_log_negative_y.png");
    
    assert!(result.is_err(), "Log Y scale should reject negative data");
}

#[test]
fn test_log_scale_rejects_zero_values() {
    let x = Vec64::from_slice(&[1.0, 2.0, 0.0, 4.0, 5.0]); // Contains zero
    let y = Vec64::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0]);
    
    let result = Plot::new()
        .title("Log Scale with Zero X")
        .xscale(Scale::Log10)
        .line(&x, &y)
        .save("test_log_zero_x.png");
    
    assert!(result.is_err(), "Log X scale should reject zero values");
}

#[test]
fn test_scatter_plot_with_log_scales() {
    let x = linspace(1.0, 100.0, 20);
    let y = x.map_f64(|x| x.ln().exp()); // Positive values
    
    let result = Plot::new()
        .title("Scatter Plot with Log Scales")
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .scatter(&x, &y)
        .save("test_scatter_log.png");
    
    assert!(result.is_ok(), "Scatter plots should work with log scales");
}

#[test]
fn test_multiple_series_with_log_scales() {
    let x = linspace(1.0, 100.0, 50);
    let y1 = x.map_f64(|x| x);           // Linear
    let y2 = x.map_f64(|x| x.powf(2.0)); // Quadratic
    let y3 = x.map_f64(|x| x.powf(0.5)); // Square root
    
    let result = Plot::new()
        .title("Multiple Series with Log Scales")
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .legend(true)
        .line_with(&x, &y1, "Linear")
        .line_with(&x, &y2, "Quadratic")
        .line_with(&x, &y3, "Square Root")
        .save("test_multi_series_log.png");
    
    assert!(result.is_ok(), "Multiple series should work with log scales");
}

#[test]
fn test_subplot_with_different_scales() {
    let x = linspace(1.0, 100.0, 50);
    let y = x.map_f64(|x| x.powf(2.0));
    
    let result = Plot::new()
        .subplots(1, 2)
        .title("Subplots with Different Scales")
        .size(1200, 400)
        .subplot(0, 0)
            .title("Linear Scale")
            .line(&x, &y)
            .build()
        .subplot(0, 1)
            .title("Log Scale")
            .xscale(Scale::Log10)
            .yscale(Scale::Log10)
            .line(&x, &y)
            .build()
        .save("test_subplot_scales.png");
    
    assert!(result.is_ok(), "Subplots with different scales should work");
}

#[test]
fn test_validation_positive_data() {
    let data = Vec64::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    
    // Should pass validation for log scale
    let x = data.clone();
    let y = data.clone();
    
    let plot = Plot::new()
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .line(&x, &y);
    
    let result = validate_plot_data(&plot);
    assert!(result.is_ok(), "Validation should pass for positive data");
}

#[test]
fn test_validation_mixed_data() {
    let x = Vec64::from_slice(&[1.0, -2.0, 3.0, 0.0]);
    let y = Vec64::from_slice(&[1.0, 4.0, 9.0, 16.0]);
    
    let plot = Plot::new()
        .xscale(Scale::Log10)
        .line(&x, &y);
    
    let result = validate_plot_data(&plot);
    assert!(result.is_err(), "Validation should fail for mixed positive/negative data");
}

#[test]
fn test_filter_positive_data() {
    let x = Vec64::from_slice(&[1.0, -2.0, 3.0, 0.0, 5.0]);
    let y = Vec64::from_slice(&[2.0, 4.0, -1.0, 8.0, 10.0]);
    
    let result = filter_positive_data(&x, &y);
    assert!(result.is_ok());
    
    let (filtered_x, filtered_y) = result.unwrap();
    
    // Only (1.0, 2.0) and (5.0, 10.0) should remain
    assert_eq!(filtered_x.data(), &[1.0, 5.0]);
    assert_eq!(filtered_y.data(), &[2.0, 10.0]);
}

#[test]
fn test_adjust_log_range() {
    // Test normal positive range
    let (min, max) = adjust_log_range(1.0, 100.0);
    assert_eq!(min, 0.1);
    assert_eq!(max, 1000.0);
    
    // Test range with negative minimum
    let (min, max) = adjust_log_range(-5.0, 100.0);
    assert_eq!(min, 1e-10);
    assert_eq!(max, 1000.0);
    
    // Test all negative range
    let (min, max) = adjust_log_range(-10.0, -1.0);
    assert_eq!(min, 1e-10);
    assert_eq!(max, 10.0);
}

#[test]
fn test_exponential_data_linear_vs_log() {
    let x = linspace(0.1, 3.0, 30);
    let y = x.map_f64(|x| (x * 2.0).exp()); // Exponential function
    
    // Linear scale version
    let linear_result = Plot::new()
        .title("Exponential Data - Linear Scale")
        .line(&x, &y)
        .save("test_exp_linear.png");
    
    // Log Y scale version  
    let log_result = Plot::new()
        .title("Exponential Data - Log Y Scale")
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save("test_exp_log_y.png");
    
    assert!(linear_result.is_ok(), "Linear scale should work");
    assert!(log_result.is_ok(), "Log Y scale should work with exponential data");
}

#[test]
fn test_power_law_data() {
    let x = linspace(1.0, 1000.0, 100);
    let y1 = x.map_f64(|x| x.powf(0.5));  // x^0.5
    let y2 = x.map_f64(|x| x.powf(1.0));  // x^1
    let y3 = x.map_f64(|x| x.powf(2.0));  // x^2
    
    let result = Plot::new()
        .title("Power Law Data - Log-Log Scale")
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .legend(true)
        .line_with(&x, &y1, "x^0.5")
        .line_with(&x, &y2, "x^1.0")
        .line_with(&x, &y3, "x^2.0")
        .save("test_power_law_loglog.png");
    
    assert!(result.is_ok(), "Power law data should work with log-log scales");
}

#[test]
fn test_scientific_notation_range() {
    let x = linspace(1e-6, 1e6, 100);
    let y = x.map_f64(|x| x * x);
    
    let result = Plot::new()
        .title("Scientific Notation Range")
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save("test_scientific_range.png");
    
    assert!(result.is_ok(), "Should handle scientific notation ranges");
}

#[test]
fn test_edge_case_very_small_values() {
    let x = linspace(1e-12, 1e-6, 50);
    let y = x.map_f64(|x| x.sqrt());
    
    let result = Plot::new()
        .title("Very Small Values")
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save("test_very_small.png");
    
    assert!(result.is_ok(), "Should handle very small positive values");
}

#[test]
fn test_edge_case_very_large_values() {
    let x = linspace(1e6, 1e12, 50);
    let y = x.map_f64(|x| x / 1e6);
    
    let result = Plot::new()
        .title("Very Large Values")
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save("test_very_large.png");
    
    assert!(result.is_ok(), "Should handle very large values");
}