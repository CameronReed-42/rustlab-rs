# RustLab AI-Optimized Documentation Template

## Purpose
This template ensures all RustLab APIs have documentation that enables AI code generators (GitHub Copilot, Claude, Cursor, etc.) to generate correct, efficient code with minimal hallucination.

---

## Template Structure

Every public function, method, and type should follow this documentation pattern:

```rust
/// [ONE-LINE SUMMARY]
/// 
/// # Mathematical Specification
/// [FORMAL MATHEMATICAL NOTATION IN ASCII]
/// 
/// # Dimensions
/// - Input: [PRECISE DIMENSION REQUIREMENTS]
/// - Output: [EXACT OUTPUT DIMENSIONS]
/// 
/// # Complexity
/// - Time: O([COMPLEXITY])
/// - Space: O([COMPLEXITY])
/// 
/// # For AI Code Generation
/// - [KEY CONSTRAINT 1]
/// - [KEY CONSTRAINT 2]
/// - Common uses: [TYPICAL USE CASES]
/// 
/// # Example
/// ```
/// [MINIMAL WORKING EXAMPLE]
/// ```
/// 
/// # Errors
/// - `[ERROR_TYPE]`: [WHEN IT OCCURS AND HOW TO FIX]
/// 
/// # See Also
/// - [`related_function`]: [RELATIONSHIP]
```

---

## Section Guidelines

### 1. One-Line Summary
- Start with an action verb (Compute, Calculate, Transform, Apply, etc.)
- Be specific about what the function does
- Avoid implementation details

**Good:** "Compute the dot product of two vectors"
**Bad:** "Dot product implementation using SIMD"

### 2. Mathematical Specification
- Use standard ASCII mathematical notation
- Define all variables clearly
- Include the mathematical formula or algorithm

**Example:**
```
Given vectors u, v ∈ ℝⁿ:
dot(u, v) = Σᵢ(uᵢ × vᵢ) for i = 1..n
```

### 3. Dimensions
- Explicitly state input dimensions using standard notation
- Specify output dimensions
- Use consistent notation: (rows × cols) for matrices, (n) for vectors

**Example:**
```
- Input: A (m × n), B (n × p)
- Output: C (m × p)
```

### 4. Complexity
- Always include both time and space complexity
- Use standard Big-O notation
- Include the variable definitions (n = vector length, etc.)

### 5. For AI Code Generation
- List constraints that AI must respect
- Include common patterns and use cases
- Mention any non-obvious requirements
- Warn about common mistakes

**Example:**
```
- Vectors must have identical length
- Returns scalar value, not a vector
- Common uses: angle calculation, similarity metrics, projection
- Note: Use ^ operator for dot product in RustLab
```

### 6. Example
- Provide a complete, runnable example
- Show the most common use case
- Include any necessary imports
- Add expected output as comments

### 7. Errors
- List each possible error type
- Explain when it occurs
- Provide guidance on how to fix it

**Example:**
```
- `DimensionMismatch`: Vectors have different lengths. Ensure both vectors are the same size.
- `NumericOverflow`: Result exceeds f64 range. Consider normalizing inputs first.
```

### 8. See Also
- Link related functions
- Explain the relationship
- Help AI understand the function ecosystem

---

## Real Examples

### Example 1: Vector Dot Product

```rust
/// Compute the dot product (inner product) of two vectors
/// 
/// # Mathematical Specification
/// Given vectors u, v ∈ ℝⁿ:
/// dot(u, v) = Σᵢ(uᵢ × vᵢ) for i = 1..n
/// 
/// # Dimensions
/// - Input: self (n), other (n) where n > 0
/// - Output: scalar T
/// 
/// # Complexity
/// - Time: O(n) where n is vector length
/// - Space: O(1)
/// 
/// # For AI Code Generation
/// - Both vectors must have identical length
/// - Returns a scalar value (f64 for VectorF64)
/// - Use the ^ operator as shorthand: `v1 ^ v2`
/// - Common uses: angle calculation, projection, similarity
/// - Zero vectors return 0.0
/// 
/// # Example
/// ```
/// use rustlab::vec64;
/// 
/// let v1 = vec64![3.0, 4.0, 0.0];
/// let v2 = vec64![1.0, 2.0, 2.0];
/// let dot_product = v1.dot(&v2);  // Returns 11.0
/// // Equivalent: let dot_product = v1 ^ v2;
/// 
/// // Calculate angle between vectors
/// let cos_angle = dot_product / (v1.norm() * v2.norm());
/// ```
/// 
/// # Errors
/// - `DimensionMismatch`: Vectors have different lengths. Ensure both vectors have the same size.
/// 
/// # See Also
/// - [`norm`]: Calculate vector magnitude (often used with dot product)
/// - [`cross`]: Cross product for 3D vectors (orthogonal operation)
/// - [`outer`]: Outer product producing a matrix
pub fn dot(&self, other: &Self) -> T {
    // Implementation
}
```

### Example 2: Matrix Multiplication

```rust
/// Multiply two matrices using standard matrix multiplication
/// 
/// # Mathematical Specification
/// Given matrices A ∈ ℝᵐˣⁿ, B ∈ ℝⁿˣᵖ:
/// C = A × B where Cᵢⱼ = Σₖ(Aᵢₖ × Bₖⱼ) for k = 1..n
/// 
/// # Dimensions
/// - Input: self (m × n), other (n × p)
/// - Output: Matrix (m × p)
/// - Constraint: self.ncols == other.nrows
/// 
/// # Complexity
/// - Time: O(m × n × p) for naive algorithm
/// - Space: O(m × p) for result matrix
/// 
/// # For AI Code Generation
/// - Inner dimensions must match: (m×n) × (n×p) → (m×p)
/// - Not commutative: A×B ≠ B×A in general
/// - Use ^ operator in RustLab: `A ^ B`
/// - For element-wise multiply use *: `A * B`
/// - Common uses: linear transformations, neural network layers
/// 
/// # Example
/// ```
/// use rustlab::array64;
/// 
/// let A = array64![[1.0, 2.0], 
///                  [3.0, 4.0]];  // 2×2
/// let B = array64![[5.0, 6.0], 
///                  [7.0, 8.0]];  // 2×2
/// 
/// let C = A.matmul(&B);  // or A ^ B
/// // Result: [[19.0, 22.0], 
/// //          [43.0, 50.0]]
/// 
/// let v = vec64![1.0, 2.0];
/// let result = A ^ v;  // Matrix-vector multiplication
/// ```
/// 
/// # Errors
/// - `DimensionMismatch`: Inner dimensions don't match. 
///   Fix: Ensure A.ncols() == B.nrows() or consider transposing
/// - `MemoryAllocation`: Result matrix too large.
///   Fix: Consider using views or chunked operations
/// 
/// # See Also
/// - [`transpose`]: Transpose matrix for dimension matching
/// - [`mul`]: Element-wise multiplication (Hadamard product)
/// - [`solve`]: Solve linear system instead of computing inverse
pub fn matmul(&self, other: &Self) -> Result<Self, MathError> {
    // Implementation
}
```

### Example 3: Linear Regression Fit

```rust
/// Fit an Ordinary Least Squares linear regression model
/// 
/// # Mathematical Specification
/// Solves the optimization problem:
/// minimize ||Xβ - y||² over β ∈ ℝᵖ
/// 
/// Solution via normal equations:
/// β = (X'X)⁻¹X'y when X'X is invertible
/// 
/// # Dimensions
/// - Input: X (n × p) design matrix, y (n) response vector
/// - Output: LinearModel with coefficients β (p)
/// - Constraint: n ≥ p (more observations than features)
/// 
/// # Complexity
/// - Time: O(p³ + n×p²) using Cholesky decomposition
/// - Space: O(p²) for normal equations
/// 
/// # For AI Code Generation
/// - Requires n_samples ≥ n_features for unique solution
/// - Features should be standardized for numerical stability
/// - Automatically adds intercept unless specified otherwise
/// - For regularization use `ridge_regression` or `lasso_regression`
/// - Common uses: prediction, feature importance, trend analysis
/// 
/// # Example
/// ```
/// use rustlab::{array64, vec64};
/// 
/// // Training data: house prices
/// let features = array64![
///     [1200.0, 3.0],  // sqft, bedrooms
///     [1500.0, 4.0],
///     [1800.0, 3.0],
///     [2000.0, 4.0]
/// ];
/// let prices = vec64![250_000.0, 300_000.0, 320_000.0, 360_000.0];
/// 
/// let model = LinearRegression::fit(&features, &prices)?;
/// let coefficients = model.coefficients();  // β vector
/// let r_squared = model.r_squared();        // Model quality
/// 
/// // Make predictions
/// let new_house = array64![[1600.0, 3.0]];
/// let predicted_price = model.predict(&new_house)?;
/// ```
/// 
/// # Errors
/// - `SingularMatrix`: X'X is not invertible (multicollinearity).
///   Fix: Remove correlated features or use ridge regression
/// - `DimensionMismatch`: X.nrows() != y.len().
///   Fix: Ensure each observation in X has a corresponding y value
/// - `InsufficientData`: Fewer samples than features.
///   Fix: Collect more data or reduce features
/// 
/// # See Also
/// - [`ridge_regression`]: L2 regularized regression for multicollinearity
/// - [`standardize`]: Standardize features before fitting
/// - [`cross_validate`]: Evaluate model generalization
pub fn fit(features: &ArrayF64, targets: &VectorF64) -> Result<LinearModel, StatsError> {
    // Implementation
}
```

---

## Special Documentation Patterns

### For Operators

```rust
/// Matrix multiplication via ^ operator
/// 
/// # Mathematical Specification
/// A ^ B computes standard matrix multiplication (not element-wise)
/// 
/// # For AI Code Generation
/// - Use ^ for matrix multiplication: `C = A ^ B`
/// - Use * for element-wise: `C = A * B`
/// - This differs from NumPy where @ is matmul and * is element-wise
impl BitXor for ArrayF64 {
    // Implementation
}
```

### For Builders

```rust
/// Configure and build a gradient descent optimizer
/// 
/// # For AI Code Generation
/// - Builder pattern ensures all required parameters are set
/// - Default learning rate: 0.01
/// - Default max iterations: 1000
/// - Common pattern:
///   ```
///   let optimizer = GradientDescent::builder()
///       .learning_rate(0.001)
///       .momentum(0.9)
///       .build()?;
///   ```
pub struct GradientDescentBuilder {
    // Implementation
}
```

### For Error Types

```rust
/// Dimension mismatch between mathematical objects
/// 
/// # For AI Code Generation
/// - Occurs when matrix/vector dimensions are incompatible
/// - Error message includes expected and actual dimensions
/// - Common fixes:
///   1. Transpose one operand: `A.transpose() ^ B`
///   2. Reshape data: `vector.reshape(n, 1)`
///   3. Check data loading: ensure correct matrix construction
#[derive(Debug, Error)]
#[error("Dimension mismatch: expected {expected:?}, got {actual:?}")]
pub struct DimensionMismatch {
    pub expected: (usize, usize),
    pub actual: (usize, usize),
}
```

---

## Validation Checklist

Before committing documentation, verify:

- [ ] One-line summary is clear and action-oriented
- [ ] Mathematical specification uses standard notation
- [ ] Dimensions are explicitly stated with constraints
- [ ] Complexity analysis includes both time and space
- [ ] AI guidance section addresses common mistakes
- [ ] Example is complete and runnable
- [ ] Error cases are documented with fixes
- [ ] Related functions are cross-referenced

---

## Benefits for AI Code Generation

This documentation pattern helps AI by:

1. **Reducing Hallucination**: Explicit constraints prevent invalid operations
2. **Improving Accuracy**: Mathematical specs provide ground truth
3. **Enabling Error Recovery**: Error documentation guides fixes
4. **Supporting Learning**: Examples show idiomatic usage
5. **Building Context**: See Also links help understand the API ecosystem

---

## Migration Guide

To update existing documentation:

1. Start with most-used functions (dot, matmul, solve, fit)
2. Add "For AI Code Generation" section first (highest impact)
3. Gradually add other sections
4. Prioritize functions that AI commonly gets wrong

---

## License

This documentation template is part of the RustLab project, which is dual-licensed under the MIT and Apache 2.0 licenses. See [LICENSE.md](LICENSE.md) for full license details.

---

*This template is a living document. Update it based on observed AI code generation patterns and common mistakes.*