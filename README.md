# hodge-belief

Hodge decomposition for belief states with interpretability in Rust.

## Features

- **Hodge decomposition**: Split signals into exact + coexact + harmonic components
- **Hodge realization**: Idempotent projection, Green's function, spectral gap
- **Belief states**: Represent and decompose multi-agent belief distributions
- **Interpretability**: Understand consensus vs disagreement via decomposition
- **Stability analysis**: Analyze decomposition robustness under perturbation
- **Jacobi eigenvalues**: Accurate eigen decomposition for small dense matrices

## Usage

```rust
use hodge_belief::*;

// Build a cycle graph Laplacian
let lap = cycle_laplacian(4);

// Decompose a signal
let signal = vec![1.0, 3.0, 2.0, -1.0];
let decomp = HodgeDecomposition::decompose(&lap, &signal).unwrap();
assert!(decomp.verify_orthogonality());
assert!(decomp.verify_reconstruction());

// Hodge realization
let hodge = HodgeRealization::from_laplacian(lap);
println!("Spectral gap: {}", hodge.spectral_gap);
println!("Harmonic dim: {}", hodge.harmonic_dimension);

// Belief decomposition
let belief = BeliefState::from_vec(vec![0.4, 0.3, 0.2, 0.1]);
let decomp = DecomposedBelief::decompose(&lap, &belief).unwrap();
let interp = decomp.interpret();
println!("Consensus fraction: {}", interp.consensus_fraction);
```

## Test Count

36 tests.

## License

MIT
