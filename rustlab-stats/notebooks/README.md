# RustLab Statistics Example Notebooks

This directory contains comprehensive Jupyter notebooks demonstrating the statistical analysis capabilities of the RustLab ecosystem. These notebooks showcase real-world statistical applications using `rustlab-stats`, `rustlab-math`, and `rustlab-plotting` for complete data analysis workflows.

## 📊 Notebook Overview

### **01_descriptive_statistics_showcase.ipynb**
**Focus**: Advanced descriptive statistics and robust measures
- **Statistics Covered**: Mean, median, mode, quartiles, IQR, MAD, skewness, kurtosis
- **Robust Statistics**: Handling outliers with robust measures vs classical statistics
- **Visualizations**: Box plots, histograms, distribution plots, quantile-quantile plots
- **Real-World Data**: Financial returns, sensor measurements, biological data
- **Key Concepts**: When to use robust vs classical statistics, distribution characterization

### **02_hypothesis_testing_guide.ipynb**
**Focus**: Statistical inference and hypothesis testing
- **Parametric Tests**: One-sample t-test, two-sample t-test, paired t-test, Welch's t-test
- **Non-Parametric Tests**: Mann-Whitney U, Wilcoxon signed-rank, chi-square goodness-of-fit
- **Visualizations**: Test statistic distributions, p-value illustrations, power analysis plots
- **A/B Testing**: Complete A/B test workflow with effect size and practical significance
- **Key Concepts**: Type I/II errors, power analysis, assumption checking, test selection

### **03_correlation_analysis.ipynb**
**Focus**: Relationship analysis between variables
- **Correlation Methods**: Pearson, Spearman, Kendall correlations with assumptions
- **Covariance Analysis**: Covariance matrices and multivariate relationships
- **Visualizations**: Scatter plots with regression lines, correlation heatmaps, pair plots
- **Real-World Examples**: Economic indicators, health metrics, scientific measurements
- **Key Concepts**: Correlation vs causation, non-linear relationships, outlier effects

### **04_data_preprocessing_pipeline.ipynb**
**Focus**: Data cleaning, transformation, and normalization
- **Scaling Methods**: Z-score normalization, robust scaling, min-max scaling, unit vector
- **Outlier Detection**: Statistical methods for identifying anomalous data points
- **Visualizations**: Before/after transformation plots, scaling comparisons, outlier detection plots
- **ML Preprocessing**: Feature scaling for machine learning applications
- **Key Concepts**: When to use different scaling methods, preserving data relationships

### **05_distribution_analysis.ipynb**
**Focus**: Understanding and characterizing data distributions
- **Shape Analysis**: Skewness, kurtosis, and distribution classification
- **Quantile Analysis**: Percentiles, quartiles, and risk metrics (VaR, CVaR)
- **Visualizations**: Distribution histograms, Q-Q plots, box plots, violin plots
- **Financial Applications**: Risk analysis, portfolio assessment, return distributions
- **Key Concepts**: Distribution types, tail behavior, risk quantification

### **06_multivariate_statistics.ipynb**
**Focus**: Analysis of multidimensional datasets
- **Array Statistics**: Axis-wise operations on 2D data matrices
- **Multivariate Descriptives**: Column-wise and row-wise statistical summaries
- **Visualizations**: Correlation matrices, heatmaps, parallel coordinate plots
- **Principal Components**: Understanding variance across multiple dimensions
- **Key Concepts**: Feature relationships, dimensionality, multivariate normality

### **07_real_world_case_studies.ipynb**
**Focus**: Complete statistical analysis workflows
- **Case Study 1**: Clinical trial analysis with hypothesis testing and effect sizes
- **Case Study 2**: Financial portfolio optimization using risk metrics
- **Case Study 3**: Quality control in manufacturing with robust statistics
- **Case Study 4**: Survey data analysis with categorical and continuous variables
- **Key Concepts**: End-to-end analysis, interpretation, decision-making

## 🎯 Learning Objectives

By working through these notebooks, you will learn to:

### **Statistical Foundations**
- Choose appropriate descriptive statistics for different data types and distributions
- Understand when to use robust statistics vs classical statistics
- Interpret measures of central tendency, spread, and shape

### **Inference and Testing**
- Design and conduct hypothesis tests for various research questions
- Understand p-values, effect sizes, and statistical vs practical significance
- Check assumptions and choose appropriate parametric vs non-parametric tests

### **Data Preprocessing**
- Apply appropriate scaling and normalization techniques
- Detect and handle outliers using statistical methods
- Prepare data for machine learning and advanced analysis

### **Visualization Skills**
- Create effective statistical visualizations using rustlab-plotting
- Understand how to visualize distributions, relationships, and test results
- Build publication-ready plots for statistical reports

### **Real-World Application**
- Apply statistical methods to domain-specific problems
- Interpret results in context and communicate findings effectively
- Make data-driven decisions using statistical evidence

## 🛠️ Technical Features

### **RustLab Ecosystem Integration**
- **rustlab-math**: Foundation for all mathematical operations and data structures
- **rustlab-stats**: Advanced statistical methods and hypothesis testing
- **rustlab-plotting**: Comprehensive visualization capabilities for statistical graphics

### **Notebook Design Principles**
- **Self-Contained Cells**: Each cell is independently compilable with proper imports
- **Math-First API**: Clean, mathematical notation without verbose prefixes
- **Real-World Data**: Examples use realistic datasets and scenarios
- **Visual Learning**: Extensive plotting to illustrate statistical concepts
- **Best Practices**: Following established Rust notebook conventions

### **Performance Considerations**
- **Efficient Algorithms**: Optimized implementations for large datasets
- **Memory Management**: Zero-copy operations where possible
- **Parallel Processing**: Utilization of multi-core systems for computational statistics
- **SIMD Optimization**: Vectorized operations for numerical computations

## 📈 Visualization Highlights

These notebooks extensively use `rustlab-plotting` to create:

### **Distribution Visualizations**
- **Histograms**: Data distribution with customizable binning
- **Box Plots**: Quartile visualization with outlier detection
- **Violin Plots**: Distribution shape with density estimation
- **Q-Q Plots**: Normality assessment and distribution comparison

### **Statistical Test Visualizations**
- **Test Statistic Distributions**: Null and alternative hypothesis illustrations
- **P-Value Regions**: Visual interpretation of statistical significance
- **Power Curves**: Effect size and sample size relationships
- **Confidence Intervals**: Uncertainty quantification in estimates

### **Correlation and Relationship Plots**
- **Scatter Plots**: Bivariate relationships with regression lines
- **Correlation Matrices**: Heatmaps for multivariate relationships
- **Pair Plots**: Comprehensive bivariate analysis
- **3D Surface Plots**: Three-dimensional relationship visualization

### **Time Series and Trend Analysis**
- **Trend Lines**: Statistical trend detection and visualization
- **Moving Averages**: Smoothing and trend identification
- **Seasonal Decomposition**: Time series component analysis
- **Control Charts**: Quality control monitoring with statistical limits

## 🚀 Getting Started

### **Prerequisites**
1. **Rust Jupyter Setup**: Working installation of evcxr and Jupyter
2. **RustLab Dependencies**: rustlab-math, rustlab-stats, rustlab-plotting
3. **Basic Statistics Knowledge**: Understanding of fundamental statistical concepts

### **Quick Start**
1. Start with `01_descriptive_statistics_showcase.ipynb` for foundation concepts
2. Progress through notebooks in order for systematic learning
3. Refer to individual notebooks for specific statistical methods
4. Use `07_real_world_case_studies.ipynb` for complete analysis examples

### **Data Sources**
- **Built-in Examples**: Synthetic datasets for learning concepts
- **Real-World Data**: Anonymized datasets from finance, healthcare, and science
- **Simulation Studies**: Monte Carlo methods for statistical illustration
- **Interactive Examples**: User-defined data for experimentation

## 🔬 Statistical Methods Covered

### **Descriptive Statistics**
- Central tendency: mean, median, mode, trimmed mean
- Variability: standard deviation, variance, MAD, IQR, range
- Shape: skewness, kurtosis, moments
- Robust measures: breakdown points, influence functions

### **Inferential Statistics**
- Parametric tests: t-tests (one-sample, two-sample, paired), ANOVA
- Non-parametric tests: Mann-Whitney U, Wilcoxon, Kruskal-Wallis
- Chi-square tests: goodness-of-fit, independence
- Power analysis: sample size determination, effect size calculation

### **Correlation and Association**
- Linear correlation: Pearson product-moment correlation
- Non-parametric correlation: Spearman rank, Kendall tau
- Partial correlation: controlling for confounding variables
- Covariance: multivariate relationships

### **Data Preprocessing**
- Normalization: z-score, robust scaling, min-max
- Outlier detection: statistical methods, robust statistics
- Missing data: detection and handling strategies
- Feature scaling: preparation for machine learning

## 💡 Best Practices Demonstrated

### **Statistical Rigor**
- Assumption checking before applying statistical tests
- Effect size calculation alongside significance testing
- Multiple comparison corrections where appropriate
- Confidence interval reporting for parameter estimates

### **Reproducible Analysis**
- Seed setting for random number generation
- Clear documentation of analysis decisions
- Version control considerations for statistical workflows
- Transparent reporting of methods and assumptions

### **Visualization Excellence**
- Publication-ready statistical graphics
- Clear axis labels and legends
- Appropriate color schemes for different audiences
- Interactive elements where beneficial

### **Code Quality**
- Modular functions for reusable statistical operations
- Error handling for edge cases and invalid inputs
- Performance optimization for large datasets
- Clear variable naming and code documentation

## 📚 Additional Resources

### **Mathematical Background**
- References to statistical theory and foundations
- Links to relevant academic papers and textbooks
- Explanations of algorithms and computational methods
- Discussion of numerical stability and accuracy

### **Practical Applications**
- Industry-specific examples and use cases
- Integration with other data science tools
- Scaling considerations for production environments
- Best practices for statistical consulting and reporting

### **Further Learning**
- Advanced topics for deeper exploration
- Related machine learning and data science concepts
- Connections to experimental design and causal inference
- Resources for continued statistical education

---

## 🎯 Notebook Philosophy

These notebooks are designed to bridge the gap between theoretical statistics and practical data analysis. They emphasize:

1. **Understanding Over Memorization**: Focus on when and why to use different methods
2. **Visual Intuition**: Extensive plotting to build statistical intuition
3. **Real-World Relevance**: Examples from actual data science applications
4. **Computational Efficiency**: Leveraging Rust's performance for statistical computing
5. **Reproducible Science**: Best practices for reliable and reproducible statistical analysis

Whether you're a student learning statistics, a researcher applying statistical methods, or a data scientist building analytical workflows, these notebooks provide comprehensive coverage of statistical analysis using the RustLab ecosystem.