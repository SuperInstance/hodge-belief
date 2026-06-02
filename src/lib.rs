//! # hodge-belief
//!
//! Hodge decomposition for belief states with interpretability.
//!
//! Implements:
//! - **Hodge decomposition**: Split any signal into exact + coexact + harmonic components
//! - **Belief states**: Represent and decompose multi-agent belief distributions
//! - **Interpretability**: Understand why agents disagree by examining decomposition components
//!
//! Uses pure Rust with no external dependencies (implements sparse-aware dense linear algebra).

#![deny(unsafe_code)]

use std::fmt;

// ============================================================
// Error type
// ============================================================

#[derive(Debug, Clone, PartialEq)]
pub enum HodgeError {
    EmptyMatrix,
    SingularMatrix,
    DimensionMismatch { expected: usize, actual: usize },
    NoConvergence(usize),
}

impl fmt::Display for HodgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HodgeError::EmptyMatrix => write!(f, "empty matrix"),
            HodgeError::SingularMatrix => write!(f, "singular matrix"),
            HodgeError::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            HodgeError::NoConvergence(iters) => {
                write!(f, "no convergence after {iters} iterations")
            }
        }
    }
}

impl std::error::Error for HodgeError {}

pub type Result<T> = std::result::Result<T, HodgeError>;

// ============================================================
// Dense matrix
// ============================================================

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    pub fn from_row_slice(rows: usize, cols: usize, data: &[f64]) -> Self {
        Matrix {
            rows,
            cols,
            data: data.to_vec(),
        }
    }

    pub fn from_diag(diag: &[f64]) -> Self {
        let n = diag.len();
        let mut m = Self::zeros(n, n);
        for (i, &v) in diag.iter().enumerate() {
            m.data[i * n + i] = v;
        }
        m
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[i * self.cols + j] = val;
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        result
    }

    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows, "matrix multiply dimension mismatch");
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    pub fn scale(&self, s: f64) -> Self {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * s).collect(),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        self.data
            .chunks_exact(self.cols)
            .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
            .collect()
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn trace(&self) -> f64 {
        (0..self.rows.min(self.cols)).map(|i| self.get(i, i)).sum()
    }

    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    /// Symmetric eigenvalues via Jacobi algorithm.
    pub fn symmetric_eigenvalues(&self) -> Vec<f64> {
        let (eigenvalues, _) = self.symmetric_eigen();
        eigenvalues
    }

    /// Jacobi eigenvalue algorithm for symmetric matrices.
    /// Much more accurate than power iteration for small dense matrices.
    pub fn symmetric_eigen(&self) -> (Vec<f64>, Self) {
        let n = self.rows;
        if n == 0 {
            return (vec![], Self::zeros(0, 0));
        }

        let mut a = self.data.clone();
        let mut v = Matrix::identity(n).data;

        let max_sweeps = 100;
        let tol = 1e-12;

        for _ in 0..max_sweeps {
            // Find largest off-diagonal element
            let mut max_off = 0.0_f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    let val = a[i * n + j].abs();
                    if val > max_off {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_off < tol {
                break;
            }

            // Compute rotation
            let app = a[p * n + p];
            let aqq = a[q * n + q];
            let apq = a[p * n + q];

            let theta = if (app - aqq).abs() < 1e-30 {
                std::f64::consts::PI / 4.0
            } else {
                0.5 * (2.0 * apq / (app - aqq)).atan()
            };

            let c = theta.cos();
            let s = theta.sin();

            // Apply rotation to A
            for i in 0..n {
                if i != p && i != q {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    a[i * n + p] = c * aip + s * aiq;
                    a[p * n + i] = c * aip + s * aiq;
                    a[i * n + q] = -s * aip + c * aiq;
                    a[q * n + i] = -s * aip + c * aiq;
                }
            }

            a[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
            a[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
            a[p * n + q] = 0.0;
            a[q * n + p] = 0.0;

            // Apply rotation to eigenvectors
            for i in 0..n {
                let vip = v[i * n + p];
                let viq = v[i * n + q];
                v[i * n + p] = c * vip + s * viq;
                v[i * n + q] = -s * vip + c * viq;
            }
        }

        let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
        let eigenvectors = Matrix {
            rows: n,
            cols: n,
            data: v,
        };

        // Sort ascending
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());

        let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
        let mut sorted_eigenvectors = Matrix::zeros(n, n);
        for (col, &idx) in indices.iter().enumerate() {
            for row in 0..n {
                sorted_eigenvectors.set(row, col, eigenvectors.get(row, idx));
            }
        }

        (sorted_eigenvalues, sorted_eigenvectors)
    }
}

// ============================================================
// Vector operations
// ============================================================

pub fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

pub fn vec_scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

// ============================================================
// Graph Laplacian builders
// ============================================================

/// Build a path graph Laplacian.
pub fn path_laplacian(n: usize) -> Matrix {
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        if i > 0 {
            l.set(i, i, l.get(i, i) + 1.0);
            l.set(i, i - 1, -1.0);
        }
        if i < n - 1 {
            l.set(i, i, l.get(i, i) + 1.0);
            l.set(i, i + 1, -1.0);
        }
    }
    l
}

/// Build a cycle graph Laplacian.
pub fn cycle_laplacian(n: usize) -> Matrix {
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        l.set(i, i, 2.0);
        l.set(i, (i + 1) % n, -1.0);
        l.set(i, (i + n - 1) % n, -1.0);
    }
    l
}

/// Build a complete graph Laplacian.
pub fn complete_laplacian(n: usize) -> Matrix {
    let mut l = Matrix::zeros(n, n);
    for i in 0..n {
        l.set(i, i, (n - 1) as f64);
        for j in 0..n {
            if i != j {
                l.set(i, j, -1.0);
            }
        }
    }
    l
}

// ============================================================
// Hodge Decomposition
// ============================================================

/// Result of the Hodge decomposition of a signal on a graph.
#[derive(Debug, Clone)]
pub struct HodgeDecomposition {
    /// Original signal.
    pub signal: Vec<f64>,
    /// Exact (gradient) component: f = δα for some α.
    pub exact: Vec<f64>,
    /// Coexact (curl) component: δβ for some β.
    pub coexact: Vec<f64>,
    /// Harmonic component: in kernel of Laplacian.
    pub harmonic: Vec<f64>,
    /// The Laplacian used.
    pub laplacian: Matrix,
    /// Eigenvalues of the Laplacian.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (columns).
    pub eigenvectors: Matrix,
}

impl HodgeDecomposition {
    /// Decompose a signal using the graph Laplacian.
    ///
    /// signal = exact + coexact + harmonic
    ///
    /// where:
    /// - harmonic = projection onto kernel of L (zero eigenvalues)
    /// - exact + coexact = projection onto image of L (non-zero eigenvalues)
    ///
    /// For 0-forms on a graph:
    /// - harmonic = constant component (for connected graph)
    /// - exact = gradient component
    /// - coexact = 0 for 0-forms
    #[allow(clippy::needless_range_loop)]
    pub fn decompose(laplacian: &Matrix, signal: &[f64]) -> Result<Self> {
        let n = laplacian.rows;
        if n == 0 {
            return Err(HodgeError::EmptyMatrix);
        }
        if signal.len() != n {
            return Err(HodgeError::DimensionMismatch {
                expected: n,
                actual: signal.len(),
            });
        }

        let (eigenvalues, eigenvectors) = laplacian.symmetric_eigen();

        // Split eigenvectors into harmonic (zero eigenvalue) and non-harmonic
        let mut harmonic = vec![0.0; n];
        let mut non_harmonic = vec![0.0; n];

        for k in 0..n {
            let ev = &eigenvalues[k];
            let col = eigenvectors.column(k);
            let proj = vec_dot(signal, &col);

            if ev.abs() < 1e-6 {
                // Harmonic component
                for i in 0..n {
                    harmonic[i] += proj * col[i];
                }
            } else {
                // Non-harmonic (exact + coexact for 0-forms, this is the "exact" part)
                for i in 0..n {
                    non_harmonic[i] += proj * col[i];
                }
            }
        }

        // For 0-forms on a graph, the decomposition is:
        // signal = harmonic + exact (gradient)
        // The coexact part is 0 for 0-forms.
        // We split the non-harmonic part into exact using the Laplacian gradient
        let exact = non_harmonic.clone();
        let coexact = vec![0.0; n];

        Ok(HodgeDecomposition {
            signal: signal.to_vec(),
            exact,
            coexact,
            harmonic,
            laplacian: laplacian.clone(),
            eigenvalues,
            eigenvectors,
        })
    }

    /// Verify orthogonality: exact · harmonic ≈ 0.
    pub fn verify_orthogonality(&self) -> bool {
        let dot_eh = vec_dot(&self.exact, &self.harmonic).abs();
        let dot_ch = vec_dot(&self.coexact, &self.harmonic).abs();
        let dot_ec = vec_dot(&self.exact, &self.coexact).abs();
        let tol = self.signal.len() as f64 * 1e-6;
        dot_eh < tol && dot_ch < tol && dot_ec < tol
    }

    /// Verify reconstruction: signal ≈ exact + coexact + harmonic.
    pub fn verify_reconstruction(&self) -> bool {
        let reconstructed = vec_add(&vec_add(&self.exact, &self.coexact), &self.harmonic);
        let diff = vec_sub(&self.signal, &reconstructed);
        vec_norm(&diff) < 1e-6 * vec_norm(&self.signal).max(1.0)
    }

    /// Energy fractions.
    pub fn energy_fractions(&self) -> (f64, f64, f64) {
        let total = vec_norm(&self.signal).powi(2);
        if total < 1e-15 {
            return (0.0, 0.0, 0.0);
        }
        let exact_frac = vec_norm(&self.exact).powi(2) / total;
        let coexact_frac = vec_norm(&self.coexact).powi(2) / total;
        let harmonic_frac = vec_norm(&self.harmonic).powi(2) / total;
        (exact_frac, coexact_frac, harmonic_frac)
    }
}

// ============================================================
// Hodge Realization
// ============================================================

/// Hodge realization: e = I - ΔG, where G is the Green's function.
#[derive(Debug, Clone)]
pub struct HodgeRealization {
    pub laplacian: Matrix,
    pub greens_function: Matrix,
    pub idempotent: Matrix,
    pub harmonic_dimension: usize,
    pub spectral_gap: f64,
    pub eigenvalues: Vec<f64>,
}

impl HodgeRealization {
    /// Build from a Laplacian matrix.
    pub fn from_laplacian(laplacian: Matrix) -> Self {
        let n = laplacian.rows;
        let (eigenvalues, eigenvectors) = laplacian.symmetric_eigen();

        // Spectral gap: smallest non-zero eigenvalue
        let spectral_gap = eigenvalues
            .iter()
            .filter(|&&v| v > 1e-6)
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let spectral_gap = if spectral_gap == f64::INFINITY {
            0.0
        } else {
            spectral_gap
        };

        // Harmonic dimension
        let harmonic_dim = eigenvalues.iter().filter(|&&v| v.abs() < 1e-6).count();

        // Pseudoinverse: G = V * diag(λ⁺) * V^T
        let mut pinv_diag = vec![0.0; n];
        for (i, &val) in eigenvalues.iter().enumerate() {
            pinv_diag[i] = if val.abs() < 1e-6 { 0.0 } else { 1.0 / val };
        }
        let pinv_matrix = Matrix::from_diag(&pinv_diag);
        let greens_function = eigenvectors
            .mul(&pinv_matrix)
            .mul(&eigenvectors.transpose());

        // e = I - ΔG
        let identity = Matrix::identity(n);
        let idempotent = identity.sub(&laplacian.mul(&greens_function));

        HodgeRealization {
            laplacian,
            greens_function,
            idempotent,
            harmonic_dimension: harmonic_dim,
            spectral_gap,
            eigenvalues,
        }
    }

    /// Apply the idempotent (project onto harmonic space).
    pub fn project(&self, v: &[f64]) -> Vec<f64> {
        self.idempotent.mul_vec(v)
    }

    /// Non-harmonic projection (image of Δ).
    pub fn non_harmonic_projection(&self, v: &[f64]) -> Vec<f64> {
        vec_sub(v, &self.project(v))
    }

    /// Verify idempotence: e∘e = e.
    pub fn verify_idempotence(&self) -> bool {
        let ee = self.idempotent.mul(&self.idempotent);
        let diff = ee.sub(&self.idempotent);
        diff.data.iter().all(|x| x.abs() < 1e-6)
    }

    /// Check if a vector is harmonic.
    pub fn is_harmonic(&self, v: &[f64]) -> bool {
        let lv = self.laplacian.mul_vec(v);
        lv.iter().all(|x| x.abs() < 1e-6)
    }

    /// Transient cost at time t.
    pub fn transient_cost(&self, t: f64) -> f64 {
        (-self.spectral_gap * t).exp()
    }

    /// Get harmonic basis vectors.
    pub fn harmonic_basis(&self) -> Vec<Vec<f64>> {
        let (_, eigenvectors) = self.laplacian.symmetric_eigen();
        self.eigenvalues
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() < 1e-6)
            .map(|(k, _)| eigenvectors.column(k))
            .collect()
    }
}

// ============================================================
// Belief States
// ============================================================

/// A belief state over a graph.
#[derive(Debug, Clone)]
pub struct BeliefState {
    pub graph_size: usize,
    pub beliefs: Vec<f64>,
}

impl BeliefState {
    /// Create a uniform belief state.
    pub fn uniform(n: usize) -> Self {
        BeliefState {
            graph_size: n,
            beliefs: vec![1.0 / n as f64; n],
        }
    }

    /// Create from a vector (will be normalized).
    pub fn from_vec(beliefs: Vec<f64>) -> Self {
        let n = beliefs.len();
        let sum: f64 = beliefs.iter().sum();
        let beliefs = if sum > 1e-15 {
            beliefs.iter().map(|x| x / sum).collect()
        } else {
            beliefs
        };
        BeliefState {
            graph_size: n,
            beliefs,
        }
    }

    /// Entropy of the belief distribution.
    pub fn entropy(&self) -> f64 {
        self.beliefs
            .iter()
            .filter(|&&x| x > 1e-15)
            .map(|&x| -x * x.ln())
            .sum()
    }

    /// KL divergence from self to other.
    pub fn kl_divergence(&self, other: &BeliefState) -> f64 {
        self.beliefs
            .iter()
            .zip(other.beliefs.iter())
            .filter(|(&x, _)| x > 1e-15)
            .map(|(x, y)| {
                let y_safe = if *y > 1e-15 { *y } else { 1e-15 };
                x * (x / y_safe).ln()
            })
            .sum()
    }

    /// Total variation distance.
    pub fn total_variation(&self, other: &BeliefState) -> f64 {
        self.beliefs
            .iter()
            .zip(other.beliefs.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / 2.0
    }
}

/// Decomposed belief state with interpretability.
#[derive(Debug, Clone)]
pub struct DecomposedBelief {
    pub original: BeliefState,
    pub harmonic_belief: BeliefState,
    pub gradient_belief: BeliefState,
    pub decomposition: HodgeDecomposition,
}

impl DecomposedBelief {
    /// Decompose a belief state using the graph Laplacian.
    pub fn decompose(laplacian: &Matrix, belief: &BeliefState) -> Result<Self> {
        let decomp = HodgeDecomposition::decompose(laplacian, &belief.beliefs)?;

        // Extract harmonic and gradient components as belief states
        let harmonic_belief = BeliefState {
            graph_size: belief.graph_size,
            beliefs: decomp.harmonic.clone(),
        };

        let gradient_belief = BeliefState {
            graph_size: belief.graph_size,
            beliefs: decomp.exact.clone(),
        };

        Ok(DecomposedBelief {
            original: belief.clone(),
            harmonic_belief,
            gradient_belief,
            decomposition: decomp,
        })
    }

    /// Interpret the decomposition: what fraction of belief is consensus vs disagreement.
    pub fn interpret(&self) -> BeliefInterpretation {
        let (exact_frac, coexact_frac, harmonic_frac) = self.decomposition.energy_fractions();

        BeliefInterpretation {
            consensus_fraction: harmonic_frac,
            disagreement_fraction: exact_frac + coexact_frac,
            consensus_belief: self.harmonic_belief.beliefs.clone(),
            disagreement_pattern: self.gradient_belief.beliefs.clone(),
        }
    }
}

/// Interpretation of a decomposed belief state.
#[derive(Debug, Clone)]
pub struct BeliefInterpretation {
    /// Fraction of belief that is consensus (harmonic).
    pub consensus_fraction: f64,
    /// Fraction of belief that is disagreement (exact + coexact).
    pub disagreement_fraction: f64,
    /// The consensus belief (harmonic component).
    pub consensus_belief: Vec<f64>,
    /// The disagreement pattern (gradient component).
    pub disagreement_pattern: Vec<f64>,
}

// ============================================================
// Stability analysis
// ============================================================

/// Analyze stability of the Hodge decomposition under perturbation.
pub fn stability_analysis(
    laplacian: &Matrix,
    perturbation_scale: f64,
    num_trials: usize,
) -> StabilityReport {
    let n = laplacian.rows;
    let realization = HodgeRealization::from_laplacian(laplacian.clone());

    let mut max_harmonic_shift = 0.0_f64;
    let mut max_exact_shift = 0.0_f64;

    for trial in 0..num_trials {
        // Generate random signal
        let signal: Vec<f64> = (0..n)
            .map(|i| ((i * 7 + trial * 13 + 1) as f64).sin() * 10.0)
            .collect();

        let decomp = HodgeDecomposition::decompose(laplacian, &signal).unwrap();

        // Perturb signal
        let perturbed: Vec<f64> = signal
            .iter()
            .enumerate()
            .map(|(i, x)| x + ((i * 11 + trial * 3) as f64).cos() * perturbation_scale)
            .collect();

        let decomp_p = HodgeDecomposition::decompose(laplacian, &perturbed).unwrap();

        let h_shift = vec_norm(&vec_sub(&decomp.harmonic, &decomp_p.harmonic));
        let e_shift = vec_norm(&vec_sub(&decomp.exact, &decomp_p.exact));

        max_harmonic_shift = max_harmonic_shift.max(h_shift);
        max_exact_shift = max_exact_shift.max(e_shift);
    }

    StabilityReport {
        perturbation_scale,
        max_harmonic_shift,
        max_exact_shift,
        spectral_gap: realization.spectral_gap,
        harmonic_dimension: realization.harmonic_dimension,
    }
}

/// Stability report.
#[derive(Debug, Clone)]
pub struct StabilityReport {
    pub perturbation_scale: f64,
    pub max_harmonic_shift: f64,
    pub max_exact_shift: f64,
    pub spectral_gap: f64,
    pub harmonic_dimension: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Matrix tests ----

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::identity(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 1.0).abs() < 1e-10);
        assert!((m.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_mul_identity() {
        let a = Matrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let i = Matrix::identity(2);
        let c = a.mul(&i);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_symmetric_eigenvalues_path() {
        let lap = path_laplacian(5);
        let eigenvalues = lap.symmetric_eigenvalues();
        assert!(eigenvalues[0].abs() < 0.2, "First eigenvalue should be ~0");
        for &ev in &eigenvalues {
            assert!(ev >= -0.5, "eigenvalue {ev} should be non-negative");
        }
    }

    #[test]
    fn test_matrix_symmetric_eigenvalues_complete() {
        let lap = complete_laplacian(4);
        let eigenvalues = lap.symmetric_eigenvalues();
        // K4 eigenvalues: 0, 4, 4, 4
        assert!(eigenvalues[0].abs() < 0.5);
        assert!(eigenvalues[1] > 1.0);
    }

    // ---- Laplacian builder tests ----

    #[test]
    fn test_path_laplacian_row_sums() {
        let lap = path_laplacian(5);
        for i in 0..5 {
            let row_sum: f64 = (0..5).map(|j| lap.get(i, j)).sum();
            assert!(row_sum.abs() < 1e-10, "Row {i} sum = {row_sum}, expected 0");
        }
    }

    #[test]
    fn test_cycle_laplacian_row_sums() {
        let lap = cycle_laplacian(4);
        for i in 0..4 {
            let row_sum: f64 = (0..4).map(|j| lap.get(i, j)).sum();
            assert!(row_sum.abs() < 1e-10);
        }
    }

    #[test]
    fn test_complete_laplacian_row_sums() {
        let lap = complete_laplacian(5);
        for i in 0..5 {
            let row_sum: f64 = (0..5).map(|j| lap.get(i, j)).sum();
            assert!(row_sum.abs() < 1e-10);
        }
    }

    // ---- Hodge decomposition tests ----

    #[test]
    fn test_hodge_decomposition_orthogonality() {
        let lap = cycle_laplacian(4);
        let signal = vec![1.0, 3.0, 2.0, -1.0];
        let decomp = HodgeDecomposition::decompose(&lap, &signal).unwrap();
        assert!(
            decomp.verify_orthogonality(),
            "Exact and harmonic should be orthogonal"
        );
    }

    #[test]
    fn test_hodge_decomposition_reconstruction() {
        let lap = cycle_laplacian(4);
        let signal = vec![1.0, 3.0, 2.0, -1.0];
        let decomp = HodgeDecomposition::decompose(&lap, &signal).unwrap();
        assert!(
            decomp.verify_reconstruction(),
            "Signal should reconstruct from components"
        );
    }

    #[test]
    fn test_hodge_constant_signal_is_harmonic() {
        let lap = cycle_laplacian(4);
        let signal = vec![5.0; 4];
        let decomp = HodgeDecomposition::decompose(&lap, &signal).unwrap();
        // Constant signal should be entirely harmonic
        assert!(
            vec_norm(&decomp.exact) < 1.0,
            "Constant signal exact component should be small"
        );
    }

    #[test]
    fn test_hodge_varying_signal_has_exact() {
        let lap = path_laplacian(5);
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let decomp = HodgeDecomposition::decompose(&lap, &signal).unwrap();
        assert!(
            vec_norm(&decomp.exact) > 0.1,
            "Varying signal should have non-trivial exact component"
        );
    }

    #[test]
    fn test_hodge_energy_fractions_sum_to_one() {
        let lap = cycle_laplacian(5);
        let signal = vec![1.0, 3.0, 2.0, -1.0, 0.0];
        let decomp = HodgeDecomposition::decompose(&lap, &signal).unwrap();
        let (exact, coexact, harmonic) = decomp.energy_fractions();
        let total = exact + coexact + harmonic;
        assert!(
            (total - 1.0).abs() < 0.2,
            "Energy fractions should sum to ~1, got {total}"
        );
    }

    #[test]
    fn test_hodge_path_graph() {
        let lap = path_laplacian(5);
        let signal = vec![3.0, -1.0, 4.0, 2.0, -7.0];
        let decomp = HodgeDecomposition::decompose(&lap, &signal).unwrap();
        assert!(decomp.verify_reconstruction());
    }

    // ---- Hodge Realization tests ----

    #[test]
    fn test_hodge_realization_path() {
        let lap = path_laplacian(5);
        let hodge = HodgeRealization::from_laplacian(lap);
        assert_eq!(
            hodge.harmonic_dimension, 1,
            "Connected graph has 1 harmonic form (constant)"
        );
        assert!(hodge.spectral_gap > 0.0);
    }

    #[test]
    fn test_hodge_realization_cycle() {
        let lap = cycle_laplacian(4);
        let hodge = HodgeRealization::from_laplacian(lap);
        assert_eq!(hodge.harmonic_dimension, 1, "Cycle has 1 harmonic form");
    }

    #[test]
    fn test_hodge_realization_complete() {
        let lap = complete_laplacian(4);
        let hodge = HodgeRealization::from_laplacian(lap);
        assert_eq!(
            hodge.harmonic_dimension, 1,
            "Complete graph has 1 harmonic form"
        );
    }

    #[test]
    fn test_hodge_idempotence_path() {
        let hodge = HodgeRealization::from_laplacian(path_laplacian(5));
        assert!(hodge.verify_idempotence(), "e∘e = e should hold");
    }

    #[test]
    fn test_hodge_idempotence_cycle() {
        let hodge = HodgeRealization::from_laplacian(cycle_laplacian(4));
        assert!(hodge.verify_idempotence());
    }

    #[test]
    fn test_hodge_is_harmonic_constant() {
        let hodge = HodgeRealization::from_laplacian(cycle_laplacian(4));
        let constant = vec![1.0; 4];
        assert!(hodge.is_harmonic(&constant));
    }

    #[test]
    fn test_hodge_not_harmonic() {
        let hodge = HodgeRealization::from_laplacian(path_laplacian(5));
        let v = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!hodge.is_harmonic(&v));
    }

    #[test]
    fn test_hodge_harmonic_projection_cycle() {
        let hodge = HodgeRealization::from_laplacian(cycle_laplacian(3));
        let constant = vec![5.0; 3];
        let harm = hodge.project(&constant);
        // Constant on a cycle should be entirely harmonic
        assert!(
            vec_norm(&vec_sub(&constant, &harm)) < 1e-4,
            "Constant should be preserved as harmonic: diff = {:?}",
            vec_sub(&constant, &harm)
        );
    }

    #[test]
    fn test_hodge_project_idempotent() {
        let hodge = HodgeRealization::from_laplacian(path_laplacian(5));
        let v = vec![3.0, -1.0, 4.0, 2.0, -7.0];
        let p1 = hodge.project(&v);
        let p2 = hodge.project(&p1);
        assert!(
            vec_norm(&vec_sub(&p1, &p2)) < 1e-6,
            "Projecting twice should give same result"
        );
    }

    #[test]
    fn test_hodge_spectral_gap_path_vs_complete() {
        let path = HodgeRealization::from_laplacian(path_laplacian(10));
        let complete = HodgeRealization::from_laplacian(complete_laplacian(10));
        assert!(
            complete.spectral_gap > path.spectral_gap,
            "Complete graph spectral gap should be larger"
        );
    }

    #[test]
    fn test_transient_cost_decreases() {
        let hodge = HodgeRealization::from_laplacian(path_laplacian(5));
        assert!(hodge.transient_cost(1.0) > hodge.transient_cost(2.0));
        assert!(hodge.transient_cost(2.0) > hodge.transient_cost(5.0));
    }

    // ---- Belief state tests ----

    #[test]
    fn test_belief_uniform() {
        let b = BeliefState::uniform(4);
        assert_eq!(b.beliefs.len(), 4);
        for &x in &b.beliefs {
            assert!((x - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_belief_from_vec_normalized() {
        let b = BeliefState::from_vec(vec![1.0, 1.0, 2.0]);
        assert!((b.beliefs[0] - 0.25).abs() < 1e-10);
        assert!((b.beliefs[1] - 0.25).abs() < 1e-10);
        assert!((b.beliefs[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_belief_entropy() {
        let uniform = BeliefState::uniform(4);
        let concentrated = BeliefState::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        assert!(uniform.entropy() > concentrated.entropy());
    }

    #[test]
    fn test_belief_kl_divergence() {
        let b1 = BeliefState::from_vec(vec![0.5, 0.5]);
        let b2 = BeliefState::from_vec(vec![0.5, 0.5]);
        let kl = b1.kl_divergence(&b2);
        assert!(
            kl.abs() < 1e-10,
            "KL divergence of identical distributions should be 0"
        );
    }

    #[test]
    fn test_belief_total_variation() {
        let b1 = BeliefState::from_vec(vec![1.0, 0.0]);
        let b2 = BeliefState::from_vec(vec![0.0, 1.0]);
        let tv = b1.total_variation(&b2);
        assert!(
            (tv - 1.0).abs() < 1e-10,
            "TV distance of disjoint distributions should be 1"
        );
    }

    // ---- Decomposed belief tests ----

    #[test]
    fn test_decompose_belief() {
        let lap = cycle_laplacian(4);
        let belief = BeliefState::from_vec(vec![1.0, 3.0, 2.0, 0.5]);
        let decomp = DecomposedBelief::decompose(&lap, &belief).unwrap();
        assert!(
            decomp.decomposition.verify_reconstruction(),
            "Belief decomposition should reconstruct"
        );
    }

    #[test]
    fn test_belief_interpretation() {
        let lap = cycle_laplacian(4);
        let belief = BeliefState::from_vec(vec![1.0, 3.0, 2.0, 0.5]);
        let decomp = DecomposedBelief::decompose(&lap, &belief).unwrap();
        let interp = decomp.interpret();
        assert!(
            (interp.consensus_fraction + interp.disagreement_fraction - 1.0).abs() < 0.2,
            "Fractions should sum to ~1"
        );
    }

    // ---- Stability tests ----

    #[test]
    fn test_stability_analysis() {
        let lap = cycle_laplacian(4);
        let report = stability_analysis(&lap, 0.1, 10);
        assert!(report.max_harmonic_shift >= 0.0);
        assert!(report.max_exact_shift >= 0.0);
        assert!(report.spectral_gap > 0.0);
    }

    #[test]
    fn test_stability_small_perturbation() {
        let lap = cycle_laplacian(4);
        let report = stability_analysis(&lap, 0.01, 10);
        // Small perturbation should cause small shifts
        assert!(
            report.max_exact_shift < 5.0,
            "Small perturbation should cause small shifts"
        );
    }

    // ---- Harmonic basis test ----

    #[test]
    fn test_harmonic_basis_cycle() {
        let hodge = HodgeRealization::from_laplacian(cycle_laplacian(4));
        let basis = hodge.harmonic_basis();
        assert_eq!(basis.len(), 1, "Cycle should have 1 harmonic basis vector");
    }

    #[test]
    fn test_harmonic_basis_path() {
        let hodge = HodgeRealization::from_laplacian(path_laplacian(5));
        let basis = hodge.harmonic_basis();
        assert_eq!(basis.len(), 1, "Path graph has 1 harmonic form (constant)");
    }
}
