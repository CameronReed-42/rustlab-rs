# RustLab-Stats: Advanced Statistical Analysis - AI Documentation

## Overview

**RustLab-Stats** is a comprehensive statistical analysis library that extends the basic statistics available in rustlab-math with advanced statistical operations, hypothesis testing, and inferential statistics. This crate follows a "math-first" philosophy, emphasizing clean APIs, type-driven design, and zero-copy operations for maximum performance.

## Crate Architecture

### Design Philosophy

1. **Math-First Approach**: Clean APIs without `stat_` prefixes (leveraging Rust's module system)
2. **Type-Driven Design**: Leveraging Rust's type system for statistical inference and safety
3. **Zero-Copy Operations**: Efficient operations on existing data structures from rustlab-math
4. **Composable Design**: Operations that work naturally with the broader RustLab ecosystem
5. **Performance-Oriented**: SIMD optimizations and efficient algorithms throughout

### Module Organization

```
rustlab-stats/
├── advanced/           # Advanced descriptive statistics
│   ├── descriptive.rs  # Advanced measures (geometric mean, trimmed mean, etc.)
│   ├── quantiles.rs    # Quantiles, percentiles, quartiles, IQR
│   ├── shape.rs        # Distribution shape measures (skewness, kurtosis)
│   └── arrays.rs       # Multi-dimensional statistical operations
├── correlation.rs      # Correlation and covariance analysis
├── hypothesis.rs       # Hypothesis testing and statistical inference
├── normalization.rs    # Data normalization and standardization
├── performance.rs      # Performance monitoring and benchmarking
└── error.rs           # Error handling and result types
```

## Core Statistical Capabilities

### 1. Advanced Descriptive Statistics

#### Central Tendency Measures
- **Arithmetic Mean**: Standard mean (available in rustlab-math)
- **Geometric Mean**: nth root of product, for positive values only
  - Formula: GM = (∏xᵢ)^(1/n) = exp(∑ln(xᵢ)/n)
  - Use Case: Growth rates, ratios, multiplicative processes
- **Harmonic Mean**: Reciprocal of arithmetic mean of reciprocals
  - Formula: HM = n / ∑(1/xᵢ)
  - Use Case: Rates, speeds, price-to-earnings ratios
- **Trimmed Mean**: Mean after removing extreme values
  - Formula: Mean of middle (1-2α)×100% of sorted data
  - Use Case: Robust estimation, outlier-resistant analysis

#### Variability Measures
- **Standard Deviation**: Root mean squared deviation (rustlab-math)
- **Median Absolute Deviation (MAD)**: Robust measure of variability
  - Formula: MAD = median(|xᵢ - median(x)|)
  - Use Case: Robust scale estimation, outlier detection
- **Coefficient of Variation**: Relative variability measure
  - Formula: CV = σ/μ (standard deviation / mean)
  - Use Case: Comparing variability across different scales
- **Range**: Simple spread measure (max - min)
- **Interquartile Range**: Middle 50% spread (Q3 - Q1)

#### Distribution Shape Measures
- **Skewness**: Measure of asymmetry
  - Formula: E[(X-μ)³]/σ³
  - Interpretation: >0 right-tailed, <0 left-tailed, =0 symmetric
- **Kurtosis**: Measure of tail heaviness
  - Formula: E[(X-μ)⁴]/σ⁴ - 3 (excess kurtosis)
  - Interpretation: >0 heavy tails, <0 light tails, =0 normal-like

### 2. Quantile Analysis

#### Quantile Methods
- **Linear Interpolation**: Most common method (R's type 7 equivalent)
- **Nearest Rank**: Lower, higher, midpoint, and nearest value methods
- **Configurable**: Support for different quantile estimation approaches

#### Key Quantile Operations
- **Percentiles**: Any percentile from 0-100
- **Quartiles**: Q1 (25th), Q2 (50th/median), Q3 (75th)
- **Median**: Robust central tendency measure
- **Interquartile Range (IQR)**: Q3 - Q1, robust spread measure

### 3. Correlation Analysis

#### Correlation Methods
- **Pearson Correlation**: Linear relationship measure
  - Formula: r = Σ(xᵢ-x̄)(yᵢ-ȳ) / √[Σ(xᵢ-x̄)²Σ(yᵢ-ȳ)²]
  - Range: [-1, 1], measures linear association
  - Use Case: Linear relationships, continuous variables

- **Spearman Correlation**: Rank-based correlation
  - Formula: Pearson correlation of ranks
  - Range: [-1, 1], measures monotonic relationships
  - Use Case: Non-linear but monotonic relationships, robust to outliers

- **Kendall's Tau**: Concordance-based correlation
  - Formula: (C - D) / (C + D), where C = concordant pairs, D = discordant pairs
  - Range: [-1, 1], based on relative ordering
  - Use Case: Small samples, ties in data, robust alternative

#### Covariance Operations
- **Sample Covariance**: Uses n-1 denominator (unbiased estimator)
- **Population Covariance**: Uses n denominator (biased but consistent)
- **Cross-Covariance**: Lagged covariance for time series analysis
- **Covariance Matrix**: Full covariance matrix for multivariate data

#### Matrix Operations
- **Correlation Matrix**: Symmetric correlation matrix for all variable pairs
- **Cross-Covariance Function**: Time series cross-correlation analysis
- **Lag Analysis**: Understanding temporal relationships between variables

### 4. Hypothesis Testing

#### Parametric Tests

**One-Sample t-test**
- **Purpose**: Test if sample mean differs from hypothesized population mean
- **Assumptions**: Normality, random sampling
- **Statistic**: t = (x̄ - μ₀)/(s/√n)
- **Use Case**: Quality control, effectiveness testing

**Two-Sample t-test (Equal Variances)**
- **Purpose**: Compare means of two independent groups
- **Assumptions**: Normality, equal variances, independence
- **Statistic**: t = (x̄₁ - x̄₂)/sₚ√(1/n₁ + 1/n₂)
- **Use Case**: A/B testing, experimental comparisons

**Welch's t-test (Unequal Variances)**
- **Purpose**: Compare means when variances are unequal
- **Advantage**: Robust to unequal variances
- **Degrees of Freedom**: Satterthwaite approximation
- **Use Case**: Real-world data with different group variances

**Paired t-test**
- **Purpose**: Compare means of paired observations
- **Method**: One-sample t-test on differences
- **Use Case**: Before/after studies, matched pairs

#### Non-Parametric Tests

**Mann-Whitney U Test**
- **Purpose**: Compare distributions of two independent groups
- **Advantages**: No normality assumption, robust to outliers
- **Method**: Rank-based comparison
- **Use Case**: Skewed data, ordinal data, small samples

**Wilcoxon Signed-Rank Test**
- **Purpose**: Paired sample or one-sample median test
- **Method**: Ranks signed differences
- **Use Case**: Non-normal paired data, median testing

**Chi-Square Goodness-of-Fit**
- **Purpose**: Test if observed frequencies match expected
- **Statistic**: χ² = Σ(Oᵢ - Eᵢ)²/Eᵢ
- **Use Case**: Distribution fitting, model validation

#### Test Results Structure
```rust
pub struct TestResult {
    pub statistic: f64,           // Test statistic value
    pub p_value: f64,            // Probability of observing result
    pub df: Option<f64>,         // Degrees of freedom
    pub critical_value: Option<f64>, // Critical value at α=0.05
    pub test_name: String,       // Human-readable test name
    pub alternative: Alternative, // Type of alternative hypothesis
}
```

### 5. Error Handling and Reliability

#### Error Types
- **InsufficientData**: Not enough observations for reliable analysis
- **ConvergenceFailure**: Iterative algorithms failing to converge
- **InvalidParameters**: Parameters outside valid ranges
- **NumericalInstability**: Numerical issues during computation
- **InvalidInput**: Malformed or inappropriate data

#### Design Principles
- **Fail-Fast**: Early detection and reporting of invalid conditions
- **Informative Errors**: Detailed context for debugging
- **Structured Errors**: Programmatically accessible error information
- **Graceful Degradation**: Reasonable behavior under edge conditions

## Performance Characteristics

### Computational Complexity

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|--------|
| Basic descriptive stats | O(n) | O(1) | Single pass algorithms |
| Quantiles | O(n log n) | O(n) | Sorting required |
| Correlation | O(n) | O(1) | Once data is paired |
| Rank correlation | O(n log n) | O(n) | Ranking step |
| t-tests | O(n) | O(1) | Direct calculation |
| Mann-Whitney U | O(n log n) | O(n) | Ranking all observations |
| Covariance matrix | O(p²n) | O(p²) | p variables, n observations |

### Memory Efficiency
- **Zero-Copy Design**: Operations directly on VectorF64/VectorF32
- **In-Place Sorting**: Minimal additional allocations
- **Streaming Statistics**: Many operations require only single pass
- **SIMD Optimizations**: Vectorized operations where beneficial

### Numerical Stability
- **Robust Algorithms**: Numerically stable implementations
- **Overflow Protection**: Safe arithmetic throughout
- **Precision Management**: Appropriate use of f32 vs f64
- **Edge Case Handling**: Graceful handling of extreme values

## Integration with RustLab Ecosystem

### Seamless Data Flow
```rust
use rustlab_math::*;
use rustlab_stats::prelude::*;

// Data starts in rustlab-math
let data = vec64![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// Basic stats (rustlab-math)
let mean = data.mean();
let std = data.std(None);

// Advanced stats (rustlab-stats)  
let median = data.median();
let skew = data.skewness();
let q1 = data.quantile(0.25, None);
```

### Type System Integration
- **Consistent APIs**: Same method signatures across vector types
- **Generic Implementations**: Support for f32 and f64 throughout
- **Trait-Based Design**: Composable statistical operations
- **Error Propagation**: Consistent error handling patterns

### Performance Integration
- **Shared Memory**: No unnecessary copying between crates
- **Optimized Paths**: Efficient algorithms leveraging rustlab-math
- **Parallel Execution**: Ready for parallel processing extensions
- **Cache Efficiency**: Memory-friendly access patterns

## Usage Patterns and Examples

### Basic Statistical Analysis
```rust
use rustlab_stats::prelude::*;
use rustlab_math::vec64;

let data = vec64![12, 15, 18, 20, 22, 25, 28, 30, 32, 35];

// Descriptive statistics
let mean = data.mean();                    // 23.7
let median = data.median();                // 23.5  
let std = data.std(None);                  // 7.77
let mad = data.mad();                      // 5.93
let iqr = data.iqr();                      // 12.5
let cv = data.coefficient_of_variation();  // 0.33

// Distribution shape
let skewness = data.skewness();           // Measure asymmetry
let kurtosis = data.kurtosis();           // Measure tail heaviness
```

### Hypothesis Testing Workflow
```rust
use rustlab_stats::prelude::*;
use rustlab_math::vec64;

// Example: A/B test analysis
let control_group = vec64![23, 25, 21, 24, 26, 22, 28, 25, 24, 27];
let treatment_group = vec64![28, 30, 26, 32, 29, 31, 27, 33, 30, 29];

// Parametric test (assuming normality)
let t_test = control_group.ttest_ind(&treatment_group, Alternative::TwoSided);
println!("t-statistic: {:.3}", t_test.statistic);
println!("p-value: {:.3}", t_test.p_value);

// Non-parametric alternative (no normality assumption)
let mann_whitney = control_group.mannwhitneyu(&treatment_group, Alternative::TwoSided);
println!("U-statistic: {:.3}", mann_whitney.statistic);
println!("p-value: {:.3}", mann_whitney.p_value);
```

### Correlation Analysis
```rust
use rustlab_stats::prelude::*;
use rustlab_math::vec64;

let height = vec64![165, 170, 175, 180, 185, 190, 195];
let weight = vec64![60, 68, 75, 80, 85, 92, 98];

// Different correlation measures
let pearson = height.pearson_correlation(&weight);      // Linear relationship
let spearman = height.spearman_correlation(&weight);    // Monotonic relationship  
let kendall = height.kendall_tau(&weight);              // Concordance-based

println!("Pearson r: {:.3}", pearson);   // ~0.99 (strong linear)
println!("Spearman ρ: {:.3}", spearman); // ~1.00 (perfect monotonic)
println!("Kendall τ: {:.3}", kendall);   // ~1.00 (perfect concordance)

// Covariance analysis
let covariance = height.covariance(&weight);
println!("Covariance: {:.2}", covariance);
```

### Advanced Quantile Analysis
```rust
use rustlab_stats::prelude::*;
use rustlab_math::vec64;

let exam_scores = vec64![
    45, 52, 58, 63, 68, 72, 75, 78, 82, 85, 
    88, 90, 92, 94, 95, 96, 97, 98, 99
];

// Quartile analysis
let (q1, q2, q3) = exam_scores.quartiles();
let iqr = exam_scores.iqr();

println!("Q1 (25th percentile): {}", q1);  // Bottom 25%
println!("Q2 (median): {}", q2);           // Middle value
println!("Q3 (75th percentile): {}", q3);  // Top 25%  
println!("IQR: {}", iqr);                  // Middle 50% spread

// Custom percentiles
let p90 = exam_scores.percentile(90.0, None);  // Top 10%
let p10 = exam_scores.percentile(10.0, None);  // Bottom 10%

// Different quantile methods
let median_linear = exam_scores.quantile(0.5, Some(QuantileMethod::Linear));
let median_nearest = exam_scores.quantile(0.5, Some(QuantileMethod::Nearest));
```

## Best Practices and Guidelines

### When to Use Each Statistical Method

#### Descriptive Statistics
- **Mean vs Median**: Use median for skewed data, mean for symmetric data
- **Standard Deviation vs MAD**: Use MAD for robust analysis with outliers
- **Geometric Mean**: For growth rates, ratios, multiplicative processes
- **Trimmed Mean**: When outliers are present but you want parametric-like estimates

#### Correlation Analysis
- **Pearson**: Linear relationships, normal data, continuous variables
- **Spearman**: Monotonic relationships, ordinal data, non-normal distributions
- **Kendall**: Small samples, many ties, robust alternative to Spearman

#### Hypothesis Testing
- **Parametric tests**: When assumptions are met (normality, homoscedasticity)
- **Non-parametric tests**: Robust alternative when assumptions violated
- **Paired vs Independent**: Matched observations vs separate groups
- **One-tailed vs Two-tailed**: Directional vs non-directional hypotheses

### Performance Optimization Tips

#### Memory Efficiency
```rust
// Good: Reuse existing vectors
let stats = data.describe_stats();

// Avoid: Unnecessary copying
let data_copy = data.clone(); // Only if necessary
```

#### Numerical Stability
```rust
// Good: Check data quality first
if data.len() < 30 {
    // Use non-parametric tests
    let result = data.wilcoxon(None, Alternative::TwoSided);
} else {
    // Parametric tests for larger samples
    let result = data.ttest_1samp(0.0, Alternative::TwoSided);
}
```

### Error Handling Patterns

#### Graceful Degradation
```rust
use rustlab_stats::prelude::*;

fn robust_correlation(x: &VectorF64, y: &VectorF64) -> f64 {
    // Try Pearson first
    if x.len() >= 10 && y.len() >= 10 {
        x.pearson_correlation(y)
    } else {
        // Fall back to Kendall for small samples
        x.kendall_tau(y)
    }
}
```

#### Comprehensive Testing Strategy
```rust
fn comprehensive_statistical_test(group1: &VectorF64, group2: &VectorF64) -> TestResult {
    // Check assumptions
    let n1 = group1.len();
    let n2 = group2.len();
    
    if n1 < 30 || n2 < 30 {
        // Use non-parametric test
        group1.mannwhitneyu(group2, Alternative::TwoSided)
    } else {
        // Check variance equality (F-test would go here)
        // For now, use Welch's t-test (robust to unequal variances)
        group1.ttest_welch(group2, Alternative::TwoSided)
    }
}
```

## Future Extensions and Roadmap

### Planned Enhancements
1. **Distribution Fitting**: Parameter estimation for common distributions
2. **Regression Analysis**: Linear, logistic, and robust regression methods
3. **Time Series Analysis**: Autocorrelation, trend detection, seasonality
4. **Multivariate Statistics**: PCA, factor analysis, clustering
5. **Resampling Methods**: Bootstrap, permutation tests, cross-validation
6. **Bayesian Statistics**: Prior/posterior analysis, credible intervals
7. **Survival Analysis**: Kaplan-Meier estimation, Cox regression

### Integration Opportunities
- **rustlab-distributions**: Enhanced distribution support for hypothesis testing
- **rustlab-optimization**: Parameter estimation and model fitting
- **rustlab-plotting**: Statistical visualization and diagnostic plots
- **Parallel Processing**: Multi-threaded statistical computations

## Conclusion

RustLab-Stats provides a comprehensive, performance-oriented statistical analysis library that extends the RustLab ecosystem with advanced statistical capabilities. Its design emphasizes mathematical rigor, computational efficiency, and seamless integration while maintaining Rust's safety guarantees and zero-cost abstraction principles.

The crate serves as a bridge between basic mathematical operations and sophisticated statistical inference, enabling researchers, data scientists, and engineers to perform rigorous statistical analysis with confidence and performance.