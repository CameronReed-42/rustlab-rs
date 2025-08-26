//! Chart type implementations for scientific data visualization
//!
//! Provides specialized implementations for different chart types with mathematical
//! context and statistical applications. Each chart type is optimized for specific
//! data patterns and analysis workflows.
//!
//! # For AI Code Generation
//!
//! ## Chart Type Selection Guide
//! 
//! | Chart Type | Data Characteristics | Mathematical Use | Implementation |
//! |------------|---------------------|------------------|----------------|
//! | **Line** | Continuous, ordered | Functions y=f(x), time series | [`line`] |
//! | **Scatter** | Discrete points | Correlation analysis, clustering | [`scatter`] |
//! | **Bar** | Categorical data | Frequencies, comparisons | [`bar`] |
//! | **Histogram** | Statistical distributions | Probability density, EDA | [`histogram`] |
//!
//! ## Mathematical Applications
//!
//! ### Line Charts
//! - Function visualization: y = f(x)
//! - Time series analysis: signal processing
//! - Parametric curves: x(t), y(t)
//! - Phase plots and trajectories
//!
//! ### Scatter Plots  
//! - Correlation analysis: Pearson r, Spearman ρ
//! - Clustering visualization: k-means, DBSCAN
//! - Feature space exploration: PCA, t-SNE
//! - Outlier detection: statistical distances
//!
//! ### Bar Charts
//! - Frequency distributions: histograms with bins
//! - Category comparisons: ANOVA visualization
//! - Survey data: Likert scales, demographics
//! - Performance metrics: benchmarks, scores
//!
//! ### Histograms
//! - Probability density estimation
//! - Normality testing: QQ plots, Shapiro-Wilk
//! - Distribution fitting: normal, gamma, etc.
//! - Data quality assessment: skewness, kurtosis
//!
//! # Usage Examples
//!
//! ```rust
//! use rustlab_plotting::*;
//! use rustlab_math::{range, VectorMathOps, random_normal};
//!
//! // Line chart: Mathematical function
//! let x = range!(-PI => PI, 1000);
//! let y = x.sin();
//! plot(&x, &y)?;  // Smooth sine curve
//!
//! // Scatter plot: Noisy data correlation
//! let x = range!(0.0 => 10.0, 50);
//! let y = x.map(|&xi| 2.0 * xi + random_normal(0.0, 0.5));
//! scatter(&x, &y)?;  // Linear relationship with noise
//!
//! // Bar chart: Category comparison
//! let categories = vec64![1.0, 2.0, 3.0, 4.0];
//! let values = vec64![23.0, 45.0, 56.0, 78.0];
//! bar(&categories, &values)?;  // Categorical data
//!
//! // Histogram: Distribution analysis
//! let data = random_normal_vec(1000, 0.0, 1.0);  // Standard normal
//! histogram(&data, 30)?;  // 30 bins for distribution shape
//! ```
//!
//! # Statistical Integration
//!
//! ```rust
//! use rustlab_stats::*;
//! use rustlab_plotting::*;
//!
//! // Distribution analysis workflow
//! let samples = random_normal_vec(1000, 100.0, 15.0);
//!
//! // Visualize distribution
//! histogram(&samples, 50)?;
//!
//! // Statistical summary
//! let mean = samples.mean();
//! let std = samples.std();
//! let skewness = samples.skewness();
//!
//! // Overlay theoretical curve
//! let x_theory = range!(50.0 => 150.0, 200);
//! let y_theory = x_theory.map(|&x| normal_pdf(x, mean, std));
//! 
//! Plot::new()
//!     .histogram(&samples, 50)
//!     .line(&x_theory, &y_theory)
//!     .title(&format!("Normal Distribution (μ={:.1}, σ={:.1})", mean, std))
//!     .show()?;
//! ```
//!
//! # Advanced Chart Customization
//!
//! ```rust
//! // Multi-series comparison
//! Plot::new()
//!     .line(&x, &experiment_1)
//!     .line(&x, &experiment_2) 
//!     .line(&x, &theoretical_model)
//!     .legend(true)
//!     .xlabel("Time (s)")
//!     .ylabel("Response")
//!     .title("Model Validation Study")
//!     .show()?;
//!
//! // Scientific publication style
//! Plot::new()
//!     .scatter(&feature_1, &feature_2)
//!     .scientific_theme()
//!     .xlabel("Principal Component 1")
//!     .ylabel("Principal Component 2")
//!     .title("PCA Visualization")
//!     .grid(true)
//!     .show()?;
//! ```
//!
//! # Performance Considerations
//!
//! ## Data Size Guidelines
//! - **Line plots**: Optimal for 100-10,000 points
//! - **Scatter plots**: Good up to 1,000-5,000 points  
//! - **Bar charts**: Best for <100 categories
//! - **Histograms**: Efficient for any data size
//!
//! ## Memory Optimization
//! ```rust
//! // For large datasets, subsample for visualization
//! let large_data = generate_large_dataset(1_000_000);
//! let subsampled = subsample(&large_data, 5000);  // 5k points
//! scatter(&subsampled.x, &subsampled.y)?;
//!
//! // Use histograms for massive datasets
//! histogram(&large_data, 100)?;  // Binning handles size
//! ```

pub mod line;
pub mod scatter;
pub mod bar;
pub mod histogram;