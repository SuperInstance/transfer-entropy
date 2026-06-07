# transfer-entropy

**Transfer entropy computation for detecting directed information flow between time series with conditional TE and significance testing.**

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

**Transfer entropy** (TE) is a model-free measure of directed information transfer between two time series. Unlike correlation, TE captures **directional** and **time-asymmetric** dependencies. Unlike Granger causality, TE makes no assumptions about linearity or Gaussianity.

TE answers: *"How much does knowing the past of X help predict the future of Y, beyond what Y's own past already tells us?"*

Formally:

```
TE(X→Y) = Σ p(yₙ₊₁, xₙ, yₙ) · log₂( p(yₙ₊₁|xₙ,yₙ) / p(yₙ₊₁|yₙ) )
```

This crate provides efficient bin-based probability estimation for computing transfer entropy, conditional transfer entropy (controlling for confounders), and shuffle-based significance testing.

## Features

- **`TimeSeries`** — Wrapper for f64 observation sequences
- **`ProbabilityEstimator`** — Histogram-based joint/conditional probability estimation
- **`TransferEntropy`** — Compute TE(X→Y) using binned distributions
- **`ConditionalTE`** — TE(X→Y | Z) controlling for confounding variables
- **`SignificanceTest`** — Shuffle-based statistical significance testing

## Installation

```toml
[dependencies]
transfer-entropy = "0.1.0"
```

## Quick Start

```rust
use transfer_entropy::*;

// Create two time series
let x = TimeSeries::new(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
let y = TimeSeries::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

// Compute transfer entropy with 2 bins
let te = TransferEntropy::new(2);
let te_xy = te.compute(&x, &y); // TE(Y→X)
println!("TE(Y→X) = {:.4} nats", te_xy);
```

## Detecting Causal Links

```rust
use transfer_entropy::*;

// Create a causal link: y[n] predicts x[n+1]
let y_data: Vec<f64> = (0..200).map(|i| (i % 5) as f64).collect();
let mut x_data = vec![0.0; 200];
for i in 0..199 {
    x_data[i + 1] = y_data[i]; // x follows y with lag 1
}

let x = TimeSeries::new(x_data);
let y = TimeSeries::new(y_data);

// Compute TE
let te = TransferEntropy::new(6).compute(&x, &y);
assert!(te > 0.0, "Should detect information flow Y→X");

// Test significance
let test = SignificanceTest::new(100, 6);
let (z_score, p_value, significant) = test.test(&x, &y);
println!("z-score: {:.2}, p-value: {:.4}, significant: {}", z_score, p_value, significant);
```

## Conditional Transfer Entropy

When a third variable Z might explain the apparent information flow:

```rust
use transfer_entropy::*;

let x = TimeSeries::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
let y = TimeSeries::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
let z = TimeSeries::new(vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]);

// TE(Y→X | Z) — controlling for Z
let cte = ConditionalTE::new(4).compute(&x, &y, &z);
println!("Conditional TE = {:.4}", cte);
```

## Probability Estimation

```rust
use transfer_entropy::*;

let est = ProbabilityEstimator::new(10);

// Discretize continuous values
let bin = est.discretize(0.73, 0.0, 1.0);
assert_eq!(bin, 7);

// Compute range
let ts = TimeSeries::new(vec![1.0, 5.0, 3.0, 8.0]);
let (min, max) = ProbabilityEstimator::range(&ts);
assert_eq!(min, 1.0);
assert_eq!(max, 8.0);

// Build joint histogram
let x = TimeSeries::new(vec![0.0, 0.5, 1.0, 0.0, 0.5]);
let y = TimeSeries::new(vec![0.0, 0.5, 1.0, 0.0, 0.5]);
let hist = est.joint_histogram_3d(&x, &y);
```

## Methodology

### Binning

Continuous values are discretized into equal-width bins:

```
bin(v) = floor((v - min) / (max - min) * num_bins)
```

### Transfer Entropy Formula

```
TE(Y→X) = Σ p(xₙ₊₁, yₙ, xₙ) · ln(p(xₙ₊₁|yₙ, xₙ) / p(xₙ₊₁|xₙ))
```

Where:
- `p(xₙ₊₁, yₙ, xₙ)` is the joint probability of the triplet
- `p(xₙ₊₁|yₙ, xₙ)` is the conditional probability given both histories
- `p(xₙ₊₁|xₙ)` is the conditional probability given only X's history

### Significance Testing

The shuffle test creates a null distribution by:
1. Computing the actual TE value
2. Randomly permuting the source series `n_shuffles` times
3. Computing TE for each shuffled version
4. Computing the z-score and p-value from the null distribution

A result is significant at α=0.05 if fewer than 5% of shuffled TE values exceed the actual TE.

## API Reference

| Type | Key Methods | Description |
|------|-------------|-------------|
| `TimeSeries` | `new`, `len`, `get`, `data` | f64 observation container |
| `ProbabilityEstimator` | `discretize`, `joint_histogram_3d` | Histogram-based probability estimation |
| `TransferEntropy` | `compute` | TE(X→Y) computation |
| `ConditionalTE` | `compute` | TE(Y→X | Z) with conditioning |
| `SignificanceTest` | `test` | Shuffle-based significance (z-score, p-value) |

## Performance

- **TE computation**: O(n × bins³) where n is the series length
- **Significance test**: O(n_shuffles × TE cost)
- **Memory**: O(bins³) for the joint histogram

For long time series with moderate bin counts (4-16), computation is fast. The shuffle test dominates runtime due to repeated TE computation.

## Limitations

- Histogram-based estimation loses information compared to kernel methods
- Bin count selection affects results (too few = bias, too many = variance)
- Conditional TE uses a simplified proxy; a full implementation needs 4D histograms
- No automatic lag selection (user must specify the embedding dimension)

## License

MIT License. See [LICENSE](LICENSE) for details.
