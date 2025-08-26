//! Hypothesis testing and statistical inference
//!
//! This module provides comprehensive hypothesis testing capabilities including parametric and 
//! non-parametric tests for comparing populations, testing assumptions, and making statistical
//! inferences. All tests return structured results with test statistics, p-values, degrees of
//! freedom, and detailed test information.
//!
//! # Test Categories
//!
//! ## Parametric Tests
//!
//! Parametric tests assume specific distributions (usually normal) and are more powerful when
//! assumptions are met but can be misleading when assumptions are violated.
//!
//! - **One-sample t-test**: Tests if sample mean differs from hypothesized population mean
//! - **Two-sample t-test**: Compares means of two independent groups (equal variances assumed)
//! - **Welch's t-test**: Compares means without equal variance assumption (more robust)
//! - **Paired t-test**: Compares means of paired observations (matched samples)
//!
//! ## Non-parametric Tests
//!
//! Non-parametric tests make fewer distributional assumptions and are more robust to outliers
//! and non-normal data, but generally have lower statistical power.
//!
//! - **Mann-Whitney U test**: Non-parametric alternative to two-sample t-test
//! - **Wilcoxon signed-rank test**: Non-parametric alternative to paired t-test
//! - **Chi-square goodness-of-fit**: Tests if observed frequencies match expected distribution
//!
//! # Mathematical Foundations
//!
//! ## Test Statistics and P-values
//!
//! The core concept in hypothesis testing is the test statistic, which quantifies how far
//! our observed data deviates from what we'd expect under the null hypothesis. The p-value
//! represents the probability of observing a test statistic at least as extreme as the one
//! calculated, assuming the null hypothesis is true.
//!
//! ## Type I and Type II Errors
//!
//! - **Type I Error (α)**: Rejecting a true null hypothesis (false positive)
//! - **Type II Error (β)**: Failing to reject a false null hypothesis (false negative)
//! - **Power (1-β)**: Probability of correctly rejecting a false null hypothesis
//!
//! # Usage Guidelines
//!
//! ## Choosing the Right Test
//!
//! 1. **Data Distribution**: Check normality assumptions with Q-Q plots or Shapiro-Wilk test
//! 2. **Sample Size**: Large samples (n≥30) are more robust to non-normality
//! 3. **Independence**: Ensure observations are independent unless using paired tests
//! 4. **Variance Homogeneity**: Check equal variances for two-sample t-tests
//!
//! ## Best Practices
//!
//! - Always state hypotheses clearly before testing
//! - Check assumptions before choosing parametric vs non-parametric tests
//! - Consider effect size in addition to statistical significance
//! - Use appropriate corrections for multiple comparisons
//! - Report confidence intervals alongside p-values when possible

use rustlab_math::{VectorF64, VectorF32, BasicStatistics};
use crate::{StatsError, Result};

/// Result of a statistical hypothesis test
///
/// This structure encapsulates all relevant information from a hypothesis test,
/// enabling both programmatic decision-making and human interpretation of results.
///
/// # Fields
///
/// - `statistic`: The computed test statistic (t, z, U, W, χ², etc.)
/// - `p_value`: Probability of observing the test statistic under H₀
/// - `df`: Degrees of freedom for parametric tests (affects critical values)
/// - `critical_value`: Critical value at α=0.05 for comparison (when available)
/// - `test_name`: Human-readable test identifier
/// - `alternative`: Direction of the alternative hypothesis
///
/// # Interpretation
///
/// - **p_value < 0.05**: Strong evidence against null hypothesis (typically reject H₀)
/// - **p_value ∈ [0.05, 0.10]**: Weak evidence against null hypothesis (borderline)
/// - **p_value > 0.10**: Insufficient evidence to reject null hypothesis
///
/// # Example
///
/// ```rust
/// use rustlab_stats::prelude::*;
/// use rustlab_math::vec64;
///
/// let sample = vec64![12.1, 11.8, 12.5, 11.9, 12.3, 12.0, 12.2];
/// let result = sample.ttest_1samp(12.0, Alternative::TwoSided);
///
/// println!("Test: {}", result.test_name);
/// println!("t-statistic: {:.3}", result.statistic);
/// println!("p-value: {:.3}", result.p_value);
/// println!("Degrees of freedom: {:?}", result.df);
///
/// if result.p_value < 0.05 {
///     println!("Reject null hypothesis at α = 0.05");
/// } else {
///     println!("Fail to reject null hypothesis at α = 0.05");
/// }
/// ```
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test statistic value
    pub statistic: f64,
    /// P-value of the test
    pub p_value: f64,
    /// Degrees of freedom (if applicable)
    pub df: Option<f64>,
    /// Critical value at alpha=0.05 (if available)
    pub critical_value: Option<f64>,
    /// Test description
    pub test_name: String,
    /// Alternative hypothesis
    pub alternative: Alternative,
}

/// Alternative hypothesis specification
///
/// The alternative hypothesis defines what we're testing for and affects both the
/// calculation of the test statistic and the interpretation of the p-value.
///
/// # Variants
///
/// ## Two-Sided Test
/// - **Null**: μ = μ₀ (or μ₁ = μ₂ for two-sample tests)
/// - **Alternative**: μ ≠ μ₀ (or μ₁ ≠ μ₂)
/// - **Use Case**: Testing for any difference, regardless of direction
/// - **P-value**: Probability in both tails of the distribution
///
/// ## One-Sided Tests
///
/// ### Greater Than
/// - **Null**: μ ≤ μ₀ (or μ₁ ≤ μ₂)
/// - **Alternative**: μ > μ₀ (or μ₁ > μ₂)
/// - **Use Case**: Testing if treatment increases the outcome
/// - **P-value**: Probability in upper tail only
///
/// ### Less Than
/// - **Null**: μ ≥ μ₀ (or μ₁ ≥ μ₂)
/// - **Alternative**: μ < μ₀ (or μ₁ < μ₂)
/// - **Use Case**: Testing if treatment decreases the outcome
/// - **P-value**: Probability in lower tail only
///
/// # Statistical Power
///
/// One-sided tests have higher statistical power when the direction is correctly
/// specified, but cannot detect effects in the opposite direction.
///
/// # Example
///
/// ```rust
/// use rustlab_stats::prelude::*;
/// use rustlab_math::vec64;
///
/// let control = vec64![10.2, 9.8, 10.5, 10.1, 9.9];
/// let treatment = vec64![11.1, 10.8, 11.3, 10.9, 11.2];
///
/// // Two-sided: Is there any difference?
/// let two_sided = control.ttest_ind(&treatment, Alternative::TwoSided);
///
/// // One-sided: Is treatment better than control?
/// let greater = control.ttest_ind(&treatment, Alternative::Greater);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Alternative {
    /// Two-sided test (not equal)
    TwoSided,
    /// One-sided test (greater than)
    Greater,
    /// One-sided test (less than)
    Less,
}

/// Trait for parametric hypothesis tests
///
/// Parametric tests assume specific probability distributions (usually normal) for the
/// underlying population. These tests are more powerful when their assumptions are met
/// but can give misleading results when assumptions are violated.
///
/// # Assumptions
///
/// Most parametric tests assume:
/// - **Normality**: Data follows approximately normal distribution
/// - **Independence**: Observations are independent of each other
/// - **Homoscedasticity**: Equal variances (for multi-sample tests)
/// - **Random Sampling**: Data represents random sample from population
///
/// # When to Use Parametric Tests
///
/// - Large sample sizes (n ≥ 30) due to Central Limit Theorem
/// - Data appears approximately normal from histograms/Q-Q plots
/// - No extreme outliers that might skew results
/// - Need maximum statistical power to detect small effects
///
/// # Mathematical Foundation
///
/// Parametric tests use exact probability distributions to calculate p-values:
/// - **t-tests**: Use Student's t-distribution
/// - **z-tests**: Use standard normal distribution (large samples)
/// - **F-tests**: Use F-distribution (variance comparisons)
pub trait ParametricTests {
    /// One-sample t-test against hypothesized population mean
    ///
    /// Tests the null hypothesis H₀: μ = μ₀ against the specified alternative.
    /// This test determines whether a sample mean differs significantly from a
    /// hypothesized population mean when the population standard deviation is unknown.
    ///
    /// # Mathematical Details
    ///
    /// The test statistic follows a Student's t-distribution with n-1 degrees of freedom:
    /// ```text
    /// t = (x̄ - μ₀) / (s / √n)
    /// ```
    /// where:
    /// - x̄ = sample mean
    /// - μ₀ = hypothesized population mean
    /// - s = sample standard deviation
    /// - n = sample size
    ///
    /// # Assumptions
    ///
    /// 1. **Normality**: Sample drawn from normally distributed population
    /// 2. **Independence**: Observations are independent
    /// 3. **Random Sampling**: Data represents random sample
    ///
    /// # Robustness
    ///
    /// - Robust to moderate departures from normality (especially n ≥ 30)
    /// - Sensitive to extreme outliers
    /// - Central Limit Theorem provides normality for large samples
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// // Test if average height differs from 170 cm
    /// let heights = vec64![168, 172, 165, 175, 170, 173, 169, 171];
    /// let result = heights.ttest_1samp(170.0, Alternative::TwoSided);
    ///
    /// println!("t-statistic: {:.3}", result.statistic);
    /// println!("p-value: {:.3}", result.p_value);
    /// ```
    ///
    /// # Arguments
    /// * `mu` - Hypothesized population mean (μ₀)
    /// * `alternative` - Direction of alternative hypothesis
    ///
    /// # Returns
    /// TestResult with t-statistic, p-value, and degrees of freedom
    fn ttest_1samp(&self, mu: f64, alternative: Alternative) -> TestResult;
    
    /// Two-sample t-test assuming equal variances (Student's t-test)
    ///
    /// Tests the null hypothesis H₀: μ₁ = μ₂ that two independent populations have
    /// equal means. This test assumes equal population variances and pools the
    /// sample variances for a more powerful test.
    ///
    /// # Mathematical Details
    ///
    /// The test statistic follows a t-distribution with n₁ + n₂ - 2 degrees of freedom:
    /// ```text
    /// t = (x̄₁ - x̄₂) / (sₚ √(1/n₁ + 1/n₂))
    /// ```
    /// where the pooled standard deviation is:
    /// ```text
    /// sₚ = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]
    /// ```
    ///
    /// # Assumptions
    ///
    /// 1. **Normality**: Both samples from normally distributed populations
    /// 2. **Independence**: Observations within and between samples are independent
    /// 3. **Equal Variances**: σ₁² = σ₂² (homoscedasticity)
    /// 4. **Random Sampling**: Both samples are random
    ///
    /// # When to Use vs Alternatives
    ///
    /// - **Use this test**: When variances appear approximately equal
    /// - **Use Welch's t-test**: When variances are clearly unequal
    /// - **Use Mann-Whitney U**: When data is non-normal or ordinal
    ///
    /// # Variance Equality Testing
    ///
    /// Check equal variances using:
    /// - **Rule of thumb**: Larger variance < 2 × smaller variance
    /// - **Levene's test**: Formal test for equal variances
    /// - **F-test**: Ratio of sample variances (assumes normality)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// // Compare effectiveness of two teaching methods
    /// let method_a = vec64![78, 82, 76, 84, 80, 79, 85, 77];
    /// let method_b = vec64![85, 88, 82, 90, 87, 86, 84, 89];
    /// 
    /// let result = method_a.ttest_ind(&method_b, Alternative::TwoSided);
    /// println!("Difference significant? {}", result.p_value < 0.05);
    /// ```
    ///
    /// # Arguments
    /// * `other` - Second independent sample
    /// * `alternative` - Direction of alternative hypothesis
    ///
    /// # Returns
    /// TestResult with t-statistic, p-value, and pooled degrees of freedom
    fn ttest_ind(&self, other: &Self, alternative: Alternative) -> TestResult;
    
    /// Paired t-test for dependent samples
    ///
    /// Tests the null hypothesis H₀: μd = 0 where μd is the mean of paired differences.
    /// This test is used when observations are naturally paired (before/after, matched
    /// subjects, etc.) and eliminates between-subject variability for increased power.
    ///
    /// # Mathematical Details
    ///
    /// The test calculates differences d₁ = x₁ - y₁, d₂ = x₂ - y₂, ..., then performs
    /// a one-sample t-test on these differences:
    /// ```text
    /// t = d̄ / (sd / √n)
    /// ```
    /// where:
    /// - d̄ = mean of differences
    /// - sd = standard deviation of differences
    /// - n = number of pairs
    ///
    /// # Assumptions
    ///
    /// 1. **Paired Observations**: Each observation in sample 1 paired with sample 2
    /// 2. **Normal Differences**: The differences follow normal distribution
    /// 3. **Independence**: Pairs are independent of each other
    ///
    /// # Advantages over Independent t-test
    ///
    /// - **Higher Power**: Removes between-subject variability
    /// - **Controls Confounding**: Each subject serves as own control
    /// - **Smaller Sample Size**: More efficient design
    ///
    /// # Common Applications
    ///
    /// - **Before/After Studies**: Pre-treatment vs post-treatment measurements
    /// - **Matched Pairs**: Twin studies, matched case-control studies
    /// - **Repeated Measures**: Same measurement at two time points
    /// - **Cross-over Designs**: Each subject receives both treatments
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// // Blood pressure before and after medication
    /// let before = vec64![140, 138, 145, 142, 139, 144, 141];
    /// let after =  vec64![135, 132, 138, 136, 133, 139, 134];
    /// 
    /// let result = before.ttest_rel(&after, Alternative::Greater).unwrap();
    /// println!("Medication effective? {}", result.p_value < 0.05);
    /// ```
    ///
    /// # Arguments
    /// * `other` - Paired sample (must have same length)
    /// * `alternative` - Direction of alternative hypothesis
    ///
    /// # Returns
    /// TestResult with t-statistic and p-value for the differences, or error if
    /// samples have unequal lengths
    ///
    /// # Errors
    /// - Returns error if sample sizes don't match
    /// - Returns error if insufficient data (< 2 pairs)
    fn ttest_rel(&self, other: &Self, alternative: Alternative) -> Result<TestResult>;
    
    /// Welch's t-test for unequal variances (Welch's unequal variances t-test)
    ///
    /// Tests the null hypothesis H₀: μ₁ = μ₂ without assuming equal population variances.
    /// This test is more robust than the standard two-sample t-test when variances
    /// differ substantially between groups.
    ///
    /// # Mathematical Details
    ///
    /// The test statistic is:
    /// ```text
    /// t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
    /// ```
    ///
    /// The degrees of freedom use the Welch-Satterthwaite equation:
    /// ```text
    /// ν = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
    /// ```
    ///
    /// # Assumptions
    ///
    /// 1. **Normality**: Both samples from normally distributed populations
    /// 2. **Independence**: Observations within and between samples independent
    /// 3. **Random Sampling**: Both samples are random
    /// 4. **No Equal Variance Assumption**: Allows σ₁² ≠ σ₂²
    ///
    /// # Advantages over Standard t-test
    ///
    /// - **Robust to Unequal Variances**: Maintains correct Type I error rate
    /// - **General Applicability**: Can be used even when variances are equal
    /// - **Conservative**: Slightly more conservative when variances are equal
    ///
    /// # When to Use
    ///
    /// - **Default Choice**: Many statisticians recommend as default two-sample test
    /// - **Unequal Sample Sizes**: Especially important when n₁ ≠ n₂
    /// - **Unequal Variances**: When variance ratio > 2 or formal tests reject equality
    /// - **Uncertain Assumptions**: When unsure about variance equality
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// // Compare reaction times with different variability
    /// let group_a = vec64![245, 251, 248, 250, 249, 252];  // Low variability
    /// let group_b = vec64![240, 265, 238, 270, 242, 268];  // High variability
    /// 
    /// let result = group_a.ttest_welch(&group_b, Alternative::TwoSided);
    /// println!("Groups differ? {}", result.p_value < 0.05);
    /// ```
    ///
    /// # Arguments
    /// * `other` - Second independent sample
    /// * `alternative` - Direction of alternative hypothesis
    ///
    /// # Returns
    /// TestResult with t-statistic, p-value, and Welch-Satterthwaite degrees of freedom
    fn ttest_welch(&self, other: &Self, alternative: Alternative) -> TestResult;
}

/// Trait for non-parametric hypothesis tests
///
/// Non-parametric tests make fewer assumptions about the underlying population
/// distribution and are more robust to outliers and non-normal data. They generally
/// have lower statistical power than parametric tests when parametric assumptions
/// are met, but can be more powerful when assumptions are violated.
///
/// # Advantages of Non-parametric Tests
///
/// - **Distribution-Free**: No assumptions about population distribution shape
/// - **Robust to Outliers**: Less sensitive to extreme values
/// - **Ordinal Data**: Can handle ordinal (ranked) data effectively
/// - **Small Samples**: Often more appropriate for small sample sizes
/// - **Violation-Proof**: Valid even when parametric assumptions fail
///
/// # Disadvantages
///
/// - **Lower Power**: Generally less powerful than parametric tests
/// - **Less Precise**: Typically test medians rather than means
/// - **Information Loss**: Ranking loses some information about distances
/// - **Limited Extensions**: Fewer advanced modeling options
///
/// # When to Use Non-parametric Tests
///
/// 1. **Skewed Data**: Highly skewed or bimodal distributions
/// 2. **Outliers Present**: When outliers cannot be removed or transformed
/// 3. **Ordinal Data**: When data is naturally ordinal (Likert scales, rankings)
/// 4. **Small Samples**: When n < 30 and normality is questionable
/// 5. **Assumption Violations**: When parametric test assumptions are clearly violated
pub trait NonParametricTests {
    /// Mann-Whitney U test (Wilcoxon rank-sum test)
    ///
    /// Tests the null hypothesis that two independent samples come from populations
    /// with the same distribution. This is the non-parametric alternative to the
    /// two-sample t-test and compares the central tendencies of two groups without
    /// assuming normality.
    ///
    /// # Mathematical Details
    ///
    /// The test ranks all observations from both samples together, then calculates:
    /// ```text
    /// U₁ = R₁ - n₁(n₁ + 1)/2
    /// U₂ = n₁ × n₂ - U₁
    /// ```
    /// where R₁ is the sum of ranks for sample 1. The smaller of U₁ and U₂ is used
    /// as the test statistic.
    ///
    /// For large samples (n₁, n₂ > 20), the distribution approximates normal:
    /// ```text
    /// z = (U - μᵁ) / σᵁ
    /// μᵁ = n₁ × n₂ / 2
    /// σᵁ = √[n₁ × n₂ × (n₁ + n₂ + 1) / 12]
    /// ```
    ///
    /// # Assumptions
    ///
    /// 1. **Independence**: Observations within and between samples are independent
    /// 2. **Random Sampling**: Both samples are randomly selected
    /// 3. **Ordinal Scale**: Data can be meaningfully ranked
    /// 4. **Similar Shapes**: For location testing, distributions should have similar shapes
    ///
    /// # Interpretation
    ///
    /// - **Location Shift**: If distributions have similar shapes, tests difference in medians
    /// - **General Difference**: Otherwise, tests if one group tends to have larger values
    /// - **Effect Size**: Can calculate probability P(X > Y) where X ∼ Group 1, Y ∼ Group 2
    ///
    /// # Advantages over t-test
    ///
    /// - **No Normality**: Works with any continuous distribution
    /// - **Robust**: Less affected by outliers and skewness
    /// - **Ordinal Data**: Appropriate for ordered categorical data
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// // Compare customer satisfaction ratings (1-10 scale)
    /// let old_system = vec64![6, 5, 7, 5, 6, 4, 7, 5, 6, 5];
    /// let new_system = vec64![7, 8, 6, 9, 8, 7, 8, 9, 7, 8];
    /// 
    /// let result = old_system.mannwhitneyu(&new_system, Alternative::Less);
    /// println!("New system better? {}", result.p_value < 0.05);
    /// ```
    ///
    /// # Arguments
    /// * `other` - Second independent sample
    /// * `alternative` - Direction of alternative hypothesis
    ///
    /// # Returns
    /// TestResult with U-statistic and p-value (using normal approximation for large samples)
    fn mannwhitneyu(&self, other: &Self, alternative: Alternative) -> TestResult;
    
    /// Wilcoxon signed-rank test for paired samples or one-sample median test
    ///
    /// This test is the non-parametric alternative to the paired t-test or one-sample
    /// t-test. It tests whether the median of differences (paired test) or the median
    /// of a single sample differs significantly from zero.
    ///
    /// # Mathematical Details
    ///
    /// ## Paired Sample Version
    /// 1. Calculate differences: dᵢ = xᵢ - yᵢ
    /// 2. Remove zero differences
    /// 3. Rank absolute differences: R(|dᵢ|)
    /// 4. Sum ranks of positive differences: W⁺ = ∑ R(|dᵢ|) for dᵢ > 0
    ///
    /// ## One-Sample Version
    /// 1. Remove zero values
    /// 2. Rank absolute values: R(|xᵢ|)
    /// 3. Sum ranks of positive values: W⁺ = ∑ R(|xᵢ|) for xᵢ > 0
    ///
    /// ## Normal Approximation (large samples)
    /// ```text
    /// z = (W⁺ - μᵂ) / σᵂ
    /// μᵂ = n(n + 1) / 4
    /// σᵂ = √[n(n + 1)(2n + 1) / 24]
    /// ```
    ///
    /// # Assumptions
    ///
    /// 1. **Paired Data**: Observations are naturally paired (if using paired version)
    /// 2. **Symmetric Distribution**: Differences (or values) come from symmetric distribution
    /// 3. **Continuous Data**: No ties in the data (or minimal ties)
    /// 4. **Independence**: Pairs are independent of each other
    ///
    /// # Advantages over t-test
    ///
    /// - **No Normality**: Only requires symmetry, not normality
    /// - **Robust**: Less affected by outliers in the differences
    /// - **Median Focus**: Tests median rather than mean (sometimes more meaningful)
    /// - **Small Samples**: Often more appropriate for small sample sizes
    ///
    /// # Common Applications
    ///
    /// - **Before/After Studies**: When differences may not be normal
    /// - **Matched Pairs**: When parametric assumptions are questionable
    /// - **Skewed Differences**: When differences are highly skewed
    /// - **Ordinal Data**: When working with ordered categorical measurements
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustlab_stats::prelude::*;
    /// use rustlab_math::vec64;
    ///
    /// // Pain scores before and after treatment (0-10 scale)
    /// let before = vec64![8, 7, 9, 6, 8, 7, 9, 8];
    /// let after =  vec64![5, 4, 6, 3, 5, 4, 6, 5];
    /// 
    /// // Paired test
    /// let result = before.wilcoxon(Some(&after), Alternative::Greater).unwrap();
    /// println!("Treatment effective? {}", result.p_value < 0.05);
    /// 
    /// // One-sample test against median = 0
    /// let differences = vec64![3, 3, 3, 3, 3, 3, 3, 3];  // all positive
    /// let result = differences.wilcoxon(None, Alternative::Greater).unwrap();
    /// ```
    ///
    /// # Arguments
    /// * `other` - Optional paired sample. If Some, performs paired test.
    ///             If None, tests if sample median differs from 0
    /// * `alternative` - Direction of alternative hypothesis
    ///
    /// # Returns
    /// TestResult with W-statistic and p-value, or error if samples have unequal
    /// lengths (paired version) or insufficient data
    ///
    /// # Errors
    /// - Returns error if paired samples have different lengths
    /// - Returns error if all differences/values are zero
    /// - Returns error if insufficient data for meaningful test
    fn wilcoxon(&self, other: Option<&Self>, alternative: Alternative) -> Result<TestResult>;
}

/// Chi-square goodness-of-fit test
///
/// Tests whether observed frequencies follow a specified theoretical distribution.
/// This test compares observed category counts with expected counts under a null
/// hypothesis and determines if the discrepancy is statistically significant.
///
/// # Mathematical Details
///
/// The test statistic follows a chi-square distribution with k-1 degrees of freedom:
/// ```text
/// χ² = ∑[(Oᵢ - Eᵢ)² / Eᵢ]
/// ```
/// where:
/// - Oᵢ = observed frequency in category i
/// - Eᵢ = expected frequency in category i
/// - k = number of categories
///
/// # Assumptions
///
/// 1. **Independence**: Each observation classified into exactly one category
/// 2. **Random Sampling**: Data represents random sample from population
/// 3. **Expected Frequencies**: All expected frequencies ≥ 5 (rule of thumb)
/// 4. **Mutually Exclusive**: Categories are mutually exclusive and exhaustive
///
/// # Sample Size Requirements
///
/// - **Minimum**: All expected frequencies should be ≥ 1
/// - **Recommended**: All expected frequencies should be ≥ 5
/// - **Alternative**: Use exact tests or combine categories if requirements not met
///
/// # Common Applications
///
/// - **Distribution Testing**: Test if data follows normal, uniform, Poisson, etc.
/// - **Model Validation**: Check if theoretical model fits observed data
/// - **Quality Control**: Verify if production follows expected proportions
/// - **Genetics**: Test Hardy-Weinberg equilibrium, linkage analysis
///
/// # Interpretation
///
/// - **Large χ²**: Observed frequencies differ substantially from expected
/// - **Small p-value**: Strong evidence against null hypothesis (poor fit)
/// - **Large p-value**: Data consistent with theoretical distribution
///
/// # Examples
///
/// ```rust
/// use rustlab_stats::prelude::*;
///
/// // Test if die is fair (expected: equal frequencies)
/// let observed = [8.0, 12.0, 10.0, 15.0, 11.0, 14.0];  // 6 faces
/// let expected = [11.67, 11.67, 11.67, 11.67, 11.67, 11.67];  // Equal
/// 
/// let result = chi2_goodness_of_fit(&observed, &expected).unwrap();
/// println!("Die is fair? {}", result.p_value >= 0.05);
/// 
/// // Test if data follows normal distribution (after binning)
/// let observed_bins = [2.0, 8.0, 15.0, 25.0, 20.0, 18.0, 8.0, 4.0];
/// let expected_normal = [1.2, 4.8, 12.1, 21.8, 24.6, 21.8, 12.1, 1.6];
/// let result = chi2_goodness_of_fit(&observed_bins, &expected_normal).unwrap();
/// ```
///
/// # Arguments
/// * `observed` - Array of observed frequencies for each category
/// * `expected` - Array of expected frequencies under null hypothesis
///
/// # Returns
/// TestResult with chi-square statistic, degrees of freedom, and p-value
///
/// # Errors
/// - Returns error if arrays have different lengths
/// - Returns error if arrays are empty
/// - Returns error if any frequency is negative
/// - Returns error if expected frequencies are too small (< 5)
///
/// # Notes
/// Currently returns NaN for p-value as it requires chi-square distribution CDF
/// implementation. In practice, you would compare the statistic with critical
/// values from chi-square tables.
pub fn chi2_goodness_of_fit(observed: &[f64], expected: &[f64]) -> Result<TestResult> {
    if observed.len() != expected.len() {
        return Err(StatsError::InvalidInput("Observed and expected must have same length".into()));
    }
    
    if observed.is_empty() {
        return Err(StatsError::InvalidInput("Cannot perform test on empty data".into()));
    }
    
    // Check for non-negative values
    if observed.iter().any(|&x| x < 0.0) || expected.iter().any(|&x| x < 0.0) {
        return Err(StatsError::InvalidInput("Frequencies must be non-negative".into()));
    }
    
    // Check for sufficient expected frequencies (rule of thumb: all >= 5)
    if expected.iter().any(|&x| x < 5.0) {
        return Err(StatsError::InvalidInput("Expected frequencies should be >= 5 for valid chi-square test".into()));
    }
    
    let chi2_stat: f64 = observed.iter()
        .zip(expected.iter())
        .map(|(&obs, &exp)| (obs - exp).powi(2) / exp)
        .sum();
    
    let df = (observed.len() - 1) as f64;
    
    // For now, return the statistic and df. P-value calculation would require
    // the incomplete gamma function or chi-square distribution implementation
    Ok(TestResult {
        statistic: chi2_stat,
        p_value: f64::NAN, // Would need chi-square CDF implementation
        df: Some(df),
        critical_value: None,
        test_name: "Chi-square goodness-of-fit test".into(),
        alternative: Alternative::TwoSided,
    })
}

// Helper functions for statistical distributions
///
/// This module provides approximations for probability density and cumulative distribution
/// functions needed for hypothesis testing. These implementations prioritize numerical
/// stability and computational efficiency over perfect accuracy.
mod distributions {
    use std::f64::consts::{E, PI};
    
    /// Student's t-distribution CDF approximation
    ///
    /// Computes P(T ≤ t) where T follows a t-distribution with specified degrees of freedom.
    /// Uses rational approximation methods for computational efficiency.
    ///
    /// # Mathematical Background
    ///
    /// The t-distribution arises when estimating the mean of a normally distributed population
    /// with unknown variance. As degrees of freedom increase, it approaches the standard normal.
    ///
    /// # Implementation Details
    ///
    /// - **Large df (> 100)**: Uses standard normal approximation (CLT)
    /// - **Moderate df**: Uses incomplete beta function relationship
    /// - **Small df**: May be less accurate, consider exact methods for critical applications
    ///
    /// # Numerical Properties
    ///
    /// - **Domain**: t ∈ (-∞, +∞), df > 0
    /// - **Range**: [0, 1]
    /// - **Symmetry**: CDF(t, df) + CDF(-t, df) = 1
    /// - **Monotonicity**: Increasing in t for fixed df
    ///
    /// # Arguments
    /// * `t` - t-statistic value
    /// * `df` - degrees of freedom (must be positive)
    ///
    /// # Returns
    /// Cumulative probability P(T ≤ t), or NaN if df ≤ 0
    pub fn t_cdf(t: f64, df: f64) -> f64 {
        if df <= 0.0 {
            return f64::NAN;
        }
        
        // For large df, approximate with standard normal
        if df > 100.0 {
            return standard_normal_cdf(t);
        }
        
        // Use beta function approximation for t-distribution CDF
        // This is a simplified implementation - a full implementation would use
        // the incomplete beta function
        let x = df / (df + t * t);
        let beta_approx = incomplete_beta_approx(0.5 * df, 0.5, x);
        
        if t >= 0.0 {
            0.5 + 0.5 * (1.0 - beta_approx)
        } else {
            0.5 * beta_approx
        }
    }
    
    /// Standard normal CDF approximation using Abramowitz and Stegun formula
    ///
    /// Computes P(Z ≤ x) where Z ~ N(0,1) using a rational approximation that achieves
    /// high accuracy (error < 1.5 × 10⁻⁷) across the entire real line.
    ///
    /// # Mathematical Background
    ///
    /// The standard normal CDF is related to the error function:
    /// ```text
    /// Φ(x) = (1 + erf(x/√2)) / 2
    /// ```
    ///
    /// This implementation uses the complementary error function (erfc) approximation
    /// from Abramowitz and Stegun "Handbook of Mathematical Functions".
    ///
    /// # Accuracy
    ///
    /// - **Maximum Error**: < 1.5 × 10⁻⁷ for all x
    /// - **Typical Error**: < 10⁻₈ for |x| < 3
    /// - **Tail Accuracy**: Maintains precision in extreme tails
    ///
    /// # Properties
    ///
    /// - **Domain**: x ∈ (-∞, +∞)
    /// - **Range**: (0, 1)
    /// - **Symmetry**: Φ(-x) = 1 - Φ(x)
    /// - **Standard Values**: Φ(0) = 0.5, Φ(1.96) ≈ 0.975
    ///
    /// # Arguments
    /// * `x` - Standard normal deviate
    ///
    /// # Returns
    /// Cumulative probability P(Z ≤ x)
    pub fn standard_normal_cdf(x: f64) -> f64 {
        // Using error function approximation from Abramowitz and Stegun
        let a1 =  0.254829592;
        let a2 = -0.284496736;
        let a3 =  1.421413741;
        let a4 = -1.453152027;
        let a5 =  1.061405429;
        let p  =  0.3275911;
        
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        0.5 * (1.0 + sign * y)
    }
    
    /// Simplified incomplete beta function approximation
    ///
    /// Computes Iₓ(a,b) = ∫₀ˣ tᵃ⁻¹(1-t)ᵇ⁻¹ dt / B(a,b), which appears in the
    /// relationship between t-distribution and beta distribution CDFs.
    ///
    /// # Mathematical Relationship
    ///
    /// For a t-distributed random variable T with ν degrees of freedom:
    /// ```text
    /// P(T ≤ t) = 1/2 + (sign(t)/2) * Iₓ(ν/2, 1/2)
    /// ```
    /// where x = ν/(ν + t²)
    ///
    /// # Implementation Note
    ///
    /// This is a simplified approximation suitable for moderate precision needs.
    /// Production implementations should use continued fractions or series
    /// expansions for higher accuracy.
    ///
    /// # Arguments
    /// * `a` - First shape parameter (> 0)
    /// * `b` - Second shape parameter (> 0)  
    /// * `x` - Integration upper limit [0, 1]
    ///
    /// # Returns
    /// Approximate value of incomplete beta function
    fn incomplete_beta_approx(a: f64, b: f64, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        if x >= 1.0 { return 1.0; }
        
        // Very basic approximation - a proper implementation would use
        // continued fractions or series expansion
        x.powf(a) * (1.0 - x).powf(b) / beta_function(a, b)
    }
    
    /// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    ///
    /// The beta function is closely related to the gamma function and appears in
    /// the normalization constant for the beta distribution. It's essential for
    /// computing incomplete beta functions used in t-distribution calculations.
    ///
    /// # Mathematical Properties
    ///
    /// - **Symmetry**: B(a,b) = B(b,a)
    /// - **Relationship to combinations**: B(m,n) = (m-1)!(n-1)!/(m+n-1)! for integers
    /// - **Integration form**: B(a,b) = ∫₀¹ tᵃ⁻¹(1-t)ᵇ⁻¹ dt
    ///
    /// # Arguments
    /// * `a` - First parameter (> 0)
    /// * `b` - Second parameter (> 0)
    ///
    /// # Returns
    /// Value of beta function B(a,b)
    fn beta_function(a: f64, b: f64) -> f64 {
        gamma_function(a) * gamma_function(b) / gamma_function(a + b)
    }
    
    /// Gamma function approximation using Stirling's approximation
    ///
    /// Computes Γ(x) = ∫₀∞ tˣ⁻¹ e⁻ᵗ dt, the extension of factorial to real numbers.
    /// Uses Stirling's asymptotic formula for computational efficiency.
    ///
    /// # Mathematical Background
    ///
    /// Stirling's approximation:
    /// ```text
    /// Γ(x) ≈ √(2π/x) * (x/e)ˣ  as x → ∞
    /// ```
    ///
    /// # Key Properties
    ///
    /// - **Recurrence**: Γ(x) = (x-1)Γ(x-1) for x > 0
    /// - **Factorial**: Γ(n) = (n-1)! for positive integers n
    /// - **Half-integers**: Γ(1/2) = √π
    /// - **Poles**: Γ(x) → ∞ as x → 0⁻, -1⁻, -2⁻, ...
    ///
    /// # Implementation Details
    ///
    /// - **x > 1**: Direct Stirling's approximation
    /// - **0 < x ≤ 1**: Uses recurrence relation Γ(x) = Γ(x+1)/x
    /// - **x ≤ 0**: Returns infinity (poles at non-positive integers)
    ///
    /// # Accuracy
    ///
    /// Stirling's approximation becomes more accurate for larger x values.
    /// For high-precision needs, consider Lanczos approximation or series methods.
    ///
    /// # Arguments
    /// * `x` - Argument to gamma function
    ///
    /// # Returns
    /// Approximate value of Γ(x), or infinity for x ≤ 0
    fn gamma_function(x: f64) -> f64 {
        if x <= 0.0 { return f64::INFINITY; }
        
        // For x > 1, use Stirling's approximation
        if x > 1.0 {
            (2.0 * PI / x).sqrt() * (x / E).powf(x)
        } else {
            // Use recurrence relation Γ(x) = Γ(x+1)/x
            gamma_function(x + 1.0) / x
        }
    }
}

impl ParametricTests for VectorF64 {
    fn ttest_1samp(&self, mu: f64, alternative: Alternative) -> TestResult {
        let data = self.as_slice_unchecked();
        let n = data.len() as f64;
        
        if n < 2.0 {
            return TestResult {
                statistic: f64::NAN,
                p_value: f64::NAN,
                df: None,
                critical_value: None,
                test_name: "One-sample t-test".into(),
                alternative,
            };
        }
        
        let sample_mean = self.mean();
        let sample_std = self.std(None);
        let se = sample_std / n.sqrt();
        
        let t_stat = (sample_mean - mu) / se;
        let df = n - 1.0;
        
        // Calculate p-value based on alternative hypothesis
        let p_value = match alternative {
            Alternative::TwoSided => 2.0 * (1.0 - distributions::t_cdf(t_stat.abs(), df)),
            Alternative::Greater => 1.0 - distributions::t_cdf(t_stat, df),
            Alternative::Less => distributions::t_cdf(t_stat, df),
        };
        
        TestResult {
            statistic: t_stat,
            p_value,
            df: Some(df),
            critical_value: None, // Would need inverse t-distribution
            test_name: "One-sample t-test".into(),
            alternative,
        }
    }
    
    fn ttest_ind(&self, other: &Self, alternative: Alternative) -> TestResult {
        let data1 = self.as_slice_unchecked();
        let data2 = other.as_slice_unchecked();
        
        let n1 = data1.len() as f64;
        let n2 = data2.len() as f64;
        
        if n1 < 2.0 || n2 < 2.0 {
            return TestResult {
                statistic: f64::NAN,
                p_value: f64::NAN,
                df: None,
                critical_value: None,
                test_name: "Two-sample t-test".into(),
                alternative,
            };
        }
        
        let mean1 = self.mean();
        let mean2 = other.mean();
        let var1 = self.var(None);
        let var2 = other.var(None);
        
        // Pooled variance
        let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
        let se = (pooled_var * (1.0/n1 + 1.0/n2)).sqrt();
        
        let t_stat = (mean1 - mean2) / se;
        let df = n1 + n2 - 2.0;
        
        let p_value = match alternative {
            Alternative::TwoSided => 2.0 * (1.0 - distributions::t_cdf(t_stat.abs(), df)),
            Alternative::Greater => 1.0 - distributions::t_cdf(t_stat, df),
            Alternative::Less => distributions::t_cdf(t_stat, df),
        };
        
        TestResult {
            statistic: t_stat,
            p_value,
            df: Some(df),
            critical_value: None,
            test_name: "Two-sample t-test".into(),
            alternative,
        }
    }
    
    fn ttest_rel(&self, other: &Self, alternative: Alternative) -> Result<TestResult> {
        let data1 = self.as_slice_unchecked();
        let data2 = other.as_slice_unchecked();
        
        if data1.len() != data2.len() {
            return Err(StatsError::InvalidInput("Samples must have equal length for paired t-test".into()));
        }
        
        if data1.len() < 2 {
            return Err(StatsError::InvalidInput("Need at least 2 pairs for paired t-test".into()));
        }
        
        // Calculate differences
        let differences: Vec<f64> = data1.iter()
            .zip(data2.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        
        let diff_vec = VectorF64::from_slice(&differences);
        
        // Perform one-sample t-test on differences against mu=0
        Ok(diff_vec.ttest_1samp(0.0, alternative))
    }
    
    fn ttest_welch(&self, other: &Self, alternative: Alternative) -> TestResult {
        let data1 = self.as_slice_unchecked();
        let data2 = other.as_slice_unchecked();
        
        let n1 = data1.len() as f64;
        let n2 = data2.len() as f64;
        
        if n1 < 2.0 || n2 < 2.0 {
            return TestResult {
                statistic: f64::NAN,
                p_value: f64::NAN,
                df: None,
                critical_value: None,
                test_name: "Welch's t-test".into(),
                alternative,
            };
        }
        
        let mean1 = self.mean();
        let mean2 = other.mean();
        let var1 = self.var(None);
        let var2 = other.var(None);
        
        let se1 = var1 / n1;
        let se2 = var2 / n2;
        let se = (se1 + se2).sqrt();
        
        let t_stat = (mean1 - mean2) / se;
        
        // Welch-Satterthwaite equation for degrees of freedom
        let df = (se1 + se2).powi(2) / (se1.powi(2) / (n1 - 1.0) + se2.powi(2) / (n2 - 1.0));
        
        let p_value = match alternative {
            Alternative::TwoSided => 2.0 * (1.0 - distributions::t_cdf(t_stat.abs(), df)),
            Alternative::Greater => 1.0 - distributions::t_cdf(t_stat, df),
            Alternative::Less => distributions::t_cdf(t_stat, df),
        };
        
        TestResult {
            statistic: t_stat,
            p_value,
            df: Some(df),
            critical_value: None,
            test_name: "Welch's t-test".into(),
            alternative,
        }
    }
}

impl NonParametricTests for VectorF64 {
    fn mannwhitneyu(&self, other: &Self, alternative: Alternative) -> TestResult {
        let data1 = self.as_slice_unchecked();
        let data2 = other.as_slice_unchecked();
        
        let n1 = data1.len();
        let n2 = data2.len();
        
        if n1 == 0 || n2 == 0 {
            return TestResult {
                statistic: f64::NAN,
                p_value: f64::NAN,
                df: None,
                critical_value: None,
                test_name: "Mann-Whitney U test".into(),
                alternative,
            };
        }
        
        // Combine and rank all observations
        let mut combined: Vec<(f64, usize)> = Vec::with_capacity(n1 + n2);
        
        // Add first sample with group indicator 0
        for &value in data1 {
            combined.push((value, 0));
        }
        
        // Add second sample with group indicator 1
        for &value in data2 {
            combined.push((value, 1));
        }
        
        // Sort by value
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Assign ranks (handling ties with average rank)
        let mut ranks = vec![0.0; combined.len()];
        let mut i = 0;
        while i < combined.len() {
            let mut j = i;
            while j < combined.len() && combined[j].0 == combined[i].0 {
                j += 1;
            }
            
            // Average rank for tied values
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for k in i..j {
                ranks[k] = avg_rank;
            }
            i = j;
        }
        
        // Sum ranks for first group
        let r1: f64 = combined.iter()
            .zip(ranks.iter())
            .filter(|((_, group), _)| *group == 0)
            .map(|(_, &rank)| rank)
            .sum();
        
        // Calculate U statistics
        let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
        let u2 = (n1 * n2) as f64 - u1;
        
        // Use smaller U statistic
        let u_stat = u1.min(u2);
        
        // For normal approximation (large samples)
        let mean_u = (n1 * n2) as f64 / 2.0;
        let var_u = (n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0;
        let z_stat = (u1 - mean_u) / var_u.sqrt();
        
        // Calculate p-value using normal approximation
        let p_value = match alternative {
            Alternative::TwoSided => 2.0 * (1.0 - distributions::standard_normal_cdf(z_stat.abs())),
            Alternative::Greater => 1.0 - distributions::standard_normal_cdf(z_stat),
            Alternative::Less => distributions::standard_normal_cdf(z_stat),
        };
        
        TestResult {
            statistic: u_stat,
            p_value,
            df: None,
            critical_value: None,
            test_name: "Mann-Whitney U test".into(),
            alternative,
        }
    }
    
    fn wilcoxon(&self, other: Option<&Self>, alternative: Alternative) -> Result<TestResult> {
        match other {
            Some(other_vec) => {
                // Paired sample test
                let data1 = self.as_slice_unchecked();
                let data2 = other_vec.as_slice_unchecked();
                
                if data1.len() != data2.len() {
                    return Err(StatsError::InvalidInput("Samples must have equal length for paired Wilcoxon test".into()));
                }
                
                // Calculate differences and their absolute values
                let differences: Vec<f64> = data1.iter()
                    .zip(data2.iter())
                    .map(|(&a, &b)| a - b)
                    .filter(|&d| d != 0.0)  // Remove zero differences
                    .collect();
                
                if differences.is_empty() {
                    return Err(StatsError::InvalidInput("All differences are zero".into()));
                }
                
                let diff_vec = VectorF64::from_slice(&differences);
                diff_vec.wilcoxon(None, alternative)
            }
            None => {
                // One-sample test against median = 0
                let data = self.as_slice_unchecked();
                let non_zero: Vec<f64> = data.iter()
                    .copied()
                    .filter(|&x| x != 0.0)
                    .collect();
                
                if non_zero.is_empty() {
                    return Err(StatsError::InvalidInput("All values are zero".into()));
                }
                
                let n = non_zero.len();
                
                // Calculate absolute values and their ranks
                let mut abs_values: Vec<(f64, bool)> = non_zero.iter()
                    .map(|&x| (x.abs(), x > 0.0))
                    .collect();
                
                // Sort by absolute value
                abs_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                
                // Assign ranks (handling ties)
                let mut ranks = vec![0.0; abs_values.len()];
                let mut i = 0;
                while i < abs_values.len() {
                    let mut j = i;
                    while j < abs_values.len() && abs_values[j].0 == abs_values[i].0 {
                        j += 1;
                    }
                    
                    let avg_rank = (i + j + 1) as f64 / 2.0;
                    for k in i..j {
                        ranks[k] = avg_rank;
                    }
                    i = j;
                }
                
                // Sum positive ranks
                let w_plus: f64 = abs_values.iter()
                    .zip(ranks.iter())
                    .filter(|((_, is_positive), _)| *is_positive)
                    .map(|(_, &rank)| rank)
                    .sum();
                
                // Calculate test statistic and p-value
                let expected = n as f64 * (n + 1) as f64 / 4.0;
                let variance = n as f64 * (n + 1) as f64 * (2 * n + 1) as f64 / 24.0;
                let z_stat = (w_plus - expected) / variance.sqrt();
                
                let p_value = match alternative {
                    Alternative::TwoSided => 2.0 * (1.0 - distributions::standard_normal_cdf(z_stat.abs())),
                    Alternative::Greater => 1.0 - distributions::standard_normal_cdf(z_stat),
                    Alternative::Less => distributions::standard_normal_cdf(z_stat),
                };
                
                Ok(TestResult {
                    statistic: w_plus,
                    p_value,
                    df: None,
                    critical_value: None,
                    test_name: "Wilcoxon signed-rank test".into(),
                    alternative,
                })
            }
        }
    }
}

// Similar implementations for VectorF32 would follow the same pattern
impl ParametricTests for VectorF32 {
    fn ttest_1samp(&self, mu: f64, alternative: Alternative) -> TestResult {
        // Convert to f64 for calculation, then convert back
        let data_f64: Vec<f64> = self.as_slice_unchecked().iter().map(|&x| x as f64).collect();
        let vec_f64 = VectorF64::from_slice(&data_f64);
        vec_f64.ttest_1samp(mu, alternative)
    }
    
    fn ttest_ind(&self, other: &Self, alternative: Alternative) -> TestResult {
        let data1_f64: Vec<f64> = self.as_slice_unchecked().iter().map(|&x| x as f64).collect();
        let data2_f64: Vec<f64> = other.as_slice_unchecked().iter().map(|&x| x as f64).collect();
        let vec1_f64 = VectorF64::from_slice(&data1_f64);
        let vec2_f64 = VectorF64::from_slice(&data2_f64);
        vec1_f64.ttest_ind(&vec2_f64, alternative)
    }
    
    fn ttest_rel(&self, other: &Self, alternative: Alternative) -> Result<TestResult> {
        let data1_f64: Vec<f64> = self.as_slice_unchecked().iter().map(|&x| x as f64).collect();
        let data2_f64: Vec<f64> = other.as_slice_unchecked().iter().map(|&x| x as f64).collect();
        let vec1_f64 = VectorF64::from_slice(&data1_f64);
        let vec2_f64 = VectorF64::from_slice(&data2_f64);
        vec1_f64.ttest_rel(&vec2_f64, alternative)
    }
    
    fn ttest_welch(&self, other: &Self, alternative: Alternative) -> TestResult {
        let data1_f64: Vec<f64> = self.as_slice_unchecked().iter().map(|&x| x as f64).collect();
        let data2_f64: Vec<f64> = other.as_slice_unchecked().iter().map(|&x| x as f64).collect();
        let vec1_f64 = VectorF64::from_slice(&data1_f64);
        let vec2_f64 = VectorF64::from_slice(&data2_f64);
        vec1_f64.ttest_welch(&vec2_f64, alternative)
    }
}

impl NonParametricTests for VectorF32 {
    fn mannwhitneyu(&self, other: &Self, alternative: Alternative) -> TestResult {
        let data1_f64: Vec<f64> = self.as_slice_unchecked().iter().map(|&x| x as f64).collect();
        let data2_f64: Vec<f64> = other.as_slice_unchecked().iter().map(|&x| x as f64).collect();
        let vec1_f64 = VectorF64::from_slice(&data1_f64);
        let vec2_f64 = VectorF64::from_slice(&data2_f64);
        vec1_f64.mannwhitneyu(&vec2_f64, alternative)
    }
    
    fn wilcoxon(&self, other: Option<&Self>, alternative: Alternative) -> Result<TestResult> {
        let data1_f64: Vec<f64> = self.as_slice_unchecked().iter().map(|&x| x as f64).collect();
        let vec1_f64 = VectorF64::from_slice(&data1_f64);
        
        match other {
            Some(other_vec) => {
                let data2_f64: Vec<f64> = other_vec.as_slice_unchecked().iter().map(|&x| x as f64).collect();
                let vec2_f64 = VectorF64::from_slice(&data2_f64);
                vec1_f64.wilcoxon(Some(&vec2_f64), alternative)
            }
            None => vec1_f64.wilcoxon(None, alternative)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustlab_math::{VectorF64, vec64};
    
    #[test]
    fn test_one_sample_ttest() {
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = data.ttest_1samp(3.0, Alternative::TwoSided);
        
        assert_eq!(result.test_name, "One-sample t-test");
        assert_eq!(result.alternative, Alternative::TwoSided);
        assert!(result.df.is_some());
        assert!((result.df.unwrap() - 4.0).abs() < 1e-10);
        assert!(result.statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
    
    #[test]
    fn test_two_sample_ttest() {
        let data1 = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec64![2.0, 3.0, 4.0, 5.0, 6.0];
        let result = data1.ttest_ind(&data2, Alternative::TwoSided);
        
        assert_eq!(result.test_name, "Two-sample t-test");
        assert!(result.df.is_some());
        assert!((result.df.unwrap() - 8.0).abs() < 1e-10);
        assert!(result.statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
    
    #[test]
    fn test_paired_ttest() {
        let before = vec64![10.0, 12.0, 13.0, 15.0, 16.0];
        let after = vec64![12.0, 14.0, 14.0, 17.0, 18.0];
        let result = before.ttest_rel(&after, Alternative::TwoSided).unwrap();
        
        assert_eq!(result.test_name, "One-sample t-test"); // Tests differences
        assert!(result.statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
    
    #[test]
    fn test_paired_ttest_unequal_length() {
        let data1 = vec64![1.0, 2.0, 3.0];
        let data2 = vec64![1.0, 2.0];
        let result = data1.ttest_rel(&data2, Alternative::TwoSided);
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_welch_ttest() {
        let data1 = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec64![10.0, 20.0, 30.0]; // Different variance and size
        let result = data1.ttest_welch(&data2, Alternative::TwoSided);
        
        assert_eq!(result.test_name, "Welch's t-test");
        assert!(result.df.is_some());
        assert!(result.statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
    
    #[test]
    fn test_mann_whitney_u() {
        let data1 = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec64![6.0, 7.0, 8.0, 9.0, 10.0];
        let result = data1.mannwhitneyu(&data2, Alternative::TwoSided);
        
        assert_eq!(result.test_name, "Mann-Whitney U test");
        assert!(result.df.is_none());
        assert!(result.statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
    
    #[test]
    fn test_wilcoxon_one_sample() {
        let data = vec64![1.0, 2.0, -1.0, 3.0, -2.0];
        let result = data.wilcoxon(None, Alternative::TwoSided).unwrap();
        
        assert_eq!(result.test_name, "Wilcoxon signed-rank test");
        assert!(result.df.is_none());
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
    
    #[test]
    fn test_wilcoxon_paired() {
        let before = vec64![10.0, 12.0, 13.0, 15.0, 16.0];
        let after = vec64![12.0, 14.0, 14.0, 17.0, 18.0];
        let result = before.wilcoxon(Some(&after), Alternative::TwoSided).unwrap();
        
        assert_eq!(result.test_name, "Wilcoxon signed-rank test");
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }
    
    #[test]
    fn test_chi2_goodness_of_fit() {
        let observed = [16.0, 18.0, 16.0, 14.0, 12.0, 12.0];
        let expected = [16.0, 16.0, 16.0, 16.0, 16.0, 8.0];
        let result = chi2_goodness_of_fit(&observed, &expected).unwrap();
        
        assert_eq!(result.test_name, "Chi-square goodness-of-fit test");
        assert!(result.df.is_some());
        assert!((result.df.unwrap() - 5.0).abs() < 1e-10);
        assert!(result.statistic >= 0.0);
    }
    
    #[test]
    fn test_chi2_invalid_input() {
        let observed = [16.0, 18.0];
        let expected = [16.0]; // Different length
        let result = chi2_goodness_of_fit(&observed, &expected);
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_alternative_hypotheses() {
        let data = vec64![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let two_sided = data.ttest_1samp(2.0, Alternative::TwoSided);
        let greater = data.ttest_1samp(2.0, Alternative::Greater);
        let less = data.ttest_1samp(2.0, Alternative::Less);
        
        assert_eq!(two_sided.alternative, Alternative::TwoSided);
        assert_eq!(greater.alternative, Alternative::Greater);
        assert_eq!(less.alternative, Alternative::Less);
        
        // Two-sided p-value should be approximately twice the one-sided (for symmetric case)
        // This is an approximate relationship, exact equality depends on the distribution
        assert!(two_sided.p_value >= greater.p_value.min(less.p_value));
    }
}