//! Integration tests for complete log plotting workflows

use rustlab_plotting::prelude::*;
use rustlab_core::*;
use std::path::Path;

#[test]
fn test_complete_scientific_workflow() {
    // Simulate a complete scientific data analysis workflow
    
    // 1. Generate experimental data with exponential decay
    let time = linspace(0.1, 50.0, 200);
    let signal = time.map_f64(|t| (1000.0_f64 * (-t / 10.0).exp() + 0.0 * 5.0).max(1.0));
    
    // 2. Create linear scale plot
    let linear_result = Plot::new()
        .title("Experimental Data - Linear Scale")
        .xlabel("Time (s)")
        .ylabel("Signal Amplitude")
        .line(&time, &signal)
        .grid(true)
        .save("integration_test_linear.png");
    
    assert!(linear_result.is_ok());
    
    // 3. Create semi-log plot (log Y scale)
    let semilog_result = Plot::new()
        .title("Experimental Data - Semi-log Scale")
        .xlabel("Time (s)")
        .ylabel("log(Signal Amplitude)")
        .yscale(Scale::Log10)
        .line(&time, &signal)
        .grid(true)
        .save("integration_test_semilog.png");
    
    assert!(semilog_result.is_ok());
    
    // 4. Create comparison subplot
    let comparison_result = Plot::new()
        .subplots(1, 2)
        .size(1200, 400)
        .title("Data Analysis Comparison")
        .subplot(0, 0)
            .title("Linear Scale")
            .xlabel("Time (s)")
            .ylabel("Signal")
            .line(&time, &signal)
            .grid(true)
            .build()
        .subplot(0, 1)
            .title("Semi-log Scale")
            .xlabel("Time (s)")
            .ylabel("log(Signal)")
            .yscale(Scale::Log10)
            .line(&time, &signal)
            .grid(true)
            .build()
        .save("integration_test_comparison.png");
    
    assert!(comparison_result.is_ok());
}

#[test]
fn test_multi_dataset_analysis() {
    // Simulate analysis of multiple datasets with different characteristics
    let x = linspace(1.0, 100.0, 100);
    
    // Different power law relationships
    let linear_data = x.map_f64(|x| (x * 2.0 + randn(1).data()[0] * 1.0).max(0.1));
    let quadratic_data = x.map_f64(|x| (x.powf(2.0) / 10.0 + randn(1).data()[0] * 2.0).max(0.1));
    let cubic_data = x.map_f64(|x| (x.powf(3.0) / 1000.0 + randn(1).data()[0] * 5.0).max(0.1));
    
    let result = Plot::new()
        .title("Multi-Dataset Analysis - Log-Log Scale")
        .xlabel("log(Input Parameter)")
        .ylabel("log(Output Response)")
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .legend(true)
        .line_with(&x, &linear_data, "Linear Relationship")
        .line_with(&x, &quadratic_data, "Quadratic Relationship")  
        .line_with(&x, &cubic_data, "Cubic Relationship")
        .grid(true)
        .save("integration_test_multi_dataset.png");
    
    assert!(result.is_ok());
}

#[test]
fn test_frequency_analysis_workflow() {
    // Simulate frequency domain analysis
    let freq = linspace(1.0, 10000.0, 1000);
    
    // Different filter responses
    let lowpass = freq.map_f64(|f| {
        let fc = 100.0_f64;
        20.0_f64 * (1.0_f64 / (1.0_f64 + (f / fc).powi(2)).sqrt()).log10()
    });
    
    let highpass = freq.map_f64(|f| {
        let fc = 1000.0_f64;
        20.0_f64 * ((f / fc) / (1.0_f64 + (f / fc).powi(2)).sqrt()).log10()
    });
    
    let bandpass = freq.map_f64(|f| {
        let fc = 500.0_f64;
        let q = 10.0_f64;
        20.0 * (1.0 / (1.0 + q.powi(2) * (f / fc - fc / f).powi(2)).sqrt()).log10()
    });
    
    let result = Plot::new()
        .title("Filter Frequency Response Analysis")
        .xlabel("Frequency (Hz)")
        .ylabel("Magnitude (dB)")
        .xscale(Scale::Log10)
        .legend(true)
        .line_with(&freq, &lowpass, "Low-pass Filter")
        .line_with(&freq, &highpass, "High-pass Filter")
        .line_with(&freq, &bandpass, "Band-pass Filter")
        .grid(true)
        .save("integration_test_frequency_analysis.png");
    
    assert!(result.is_ok());
}

#[test]
fn test_error_handling_workflow() {
    // Test complete workflow with error handling
    
    // Create data with some negative values
    let x = linspace(-5.0, 5.0, 11);
    let y = x.map_f64(|x| x * x);
    
    // This should fail gracefully
    let error_result = Plot::new()
        .title("This Should Fail")
        .xscale(Scale::Log10)
        .line(&x, &y)
        .save("integration_test_error.png");
    
    assert!(error_result.is_err());
    
    // Verify error message is helpful
    if let Err(error) = error_result {
        let error_msg = error.to_string();
        assert!(error_msg.contains("non-positive values"));
        assert!(error_msg.contains("Logarithmic scales require all positive values"));
    }
    
    // Now create a working version with positive data
    let x_positive = linspace(1.0, 10.0, 10);
    let y_positive = x_positive.map_f64(|x| x * x);
    
    let success_result = Plot::new()
        .title("This Should Work")
        .xscale(Scale::Log10)
        .line(&x_positive, &y_positive)
        .save("integration_test_success.png");
    
    assert!(success_result.is_ok());
}

#[test]
fn test_performance_with_large_dataset() {
    // Test performance with larger datasets
    let start = std::time::Instant::now();
    
    let x = linspace(1.0, 10000.0, 10000); // 10k points
    let y = x.map_f64(|x| x.powf(1.5));
    
    let result = Plot::new()
        .title("Large Dataset Test - Log Scale")
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save("integration_test_large_dataset.png");
    
    let duration = start.elapsed();
    
    assert!(result.is_ok());
    // Should complete within reasonable time (adjust as needed)
    assert!(duration.as_secs() < 10, "Large dataset rendering took too long: {:?}", duration);
}

#[test]
fn test_mixed_plot_types_with_log_scales() {
    // Test different plot types with log scales
    let x1 = linspace(1.0, 100.0, 50);
    let y1 = x1.map_f64(|x| x.powf(2.0));
    
    let x2 = linspace(10.0, 90.0, 20);
    let y2 = x2.map_f64(|x| (x.powf(1.5) + randn(1).data()[0] * x * 0.05).max(1.0));
    
    let result = Plot::new()
        .title("Mixed Plot Types - Log Scales")
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .legend(true)
        .line_with(&x1, &y1, "Smooth Curve")
        .scatter_with(&x2, &y2, "Data Points")
        .grid(true)
        .save("integration_test_mixed_types.png");
    
    assert!(result.is_ok());
}

#[test]
fn test_themes_with_log_scales() {
    // Test different themes work with log scales
    let x = linspace(1.0, 1000.0, 100);
    let y = x.map_f64(|x| x.sqrt());
    
    // Test dark theme
    let dark_result = Plot::new()
        .title("Dark Theme with Log Scales")
        .dark_theme()
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save("integration_test_dark_theme.png");
    
    assert!(dark_result.is_ok());
    
    // Test scientific theme
    let sci_result = Plot::new()
        .title("Scientific Theme with Log Scales")
        .scientific_theme()
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save("integration_test_scientific_theme.png");
    
    assert!(sci_result.is_ok());
    
    // Test colorblind friendly theme
    let cb_result = Plot::new()
        .title("Colorblind Friendly Theme with Log Scales")
        .colorblind_friendly_theme()
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save("integration_test_colorblind_theme.png");
    
    assert!(cb_result.is_ok());
}

#[test]
fn test_axis_labels_and_formatting() {
    // Test that axis labels and formatting work correctly with log scales
    let x = linspace(1e-6, 1e6, 100);
    let y = x.map_f64(|x| x.powf(2.0));
    
    let result = Plot::new()
        .title("Axis Labels and Formatting Test")
        .xlabel("Input Parameter (μm)")
        .ylabel("Output Response (μN²)")
        .xscale(Scale::Log10)
        .yscale(Scale::Log10)
        .line(&x, &y)
        .grid(true)
        .save("integration_test_formatting.png");
    
    assert!(result.is_ok());
}

fn file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

#[test]
fn test_file_output_verification() {
    // Create a simple plot and verify the file is actually created
    let x = linspace(1.0, 10.0, 10);
    let y = x.map_f64(|x| x.powf(2.0));
    
    let filename = "integration_test_file_output.png";
    
    // Remove file if it exists
    if file_exists(filename) {
        std::fs::remove_file(filename).ok();
    }
    
    let result = Plot::new()
        .title("File Output Verification")
        .yscale(Scale::Log10)
        .line(&x, &y)
        .save(filename);
    
    assert!(result.is_ok());
    assert!(file_exists(filename), "Output file should be created");
    
    // Clean up
    std::fs::remove_file(filename).ok();
}