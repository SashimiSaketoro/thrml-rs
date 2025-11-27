# Byte-Level Diffusion for ROOTS Layer Compression

## Research Summary

This document captures research notes on using diffusion models (via the existing Langevin dynamics infrastructure) for learning compressed byte representations in the ROOTS layer.

## Key Insight: Langevin IS Diffusion

The core realization is that **Langevin dynamics and diffusion models are mathematically equivalent**:

### Forward Diffusion Process
```
x_t = x_{t-1} + sqrt(2T*dt) * noise
```
This is exactly what `thrml-samplers/langevin.rs` implements.

### Reverse Diffusion (Denoising)
```
x_{t-1} = x_t - dt * ∇E(x_t) + sqrt(2T*dt) * noise
```
The energy gradient `∇E` serves as the **score function** in score-matching literature.

### Implication
If we learn an energy function E over byte representations, we automatically get:
1. A way to compress (diffuse forward = add noise)
2. A way to decompress (reverse = denoise via energy gradient)

## ROOTS Layer Design

The ROOTS layer serves as the innermost shell of the sphere, containing ultra-compressed byte-level representations.

### Purpose
1. **Coarse Index**: Provides quick relevance assessment without loading full embeddings
2. **Hierarchical Navigation**: First point of contact for multi-cone navigation
3. **Memory Efficiency**: ~3000:1 compression ratio vs full embeddings

### Compression Strategy (from plan)

For each partition k in ROOTS:
```
centroid_k = (1/|P_k|) * sum_{i in P_k} embed_i

ngram_dist_k = histogram of byte unigrams in partition k

prominence_stats_k = {
    mean: mean(prominence[P_k]),
    std: std(prominence[P_k]),
    max: max(prominence[P_k])
}

radius_range_k = [min(r[P_k]), max(r[P_k])]
```

### Index Size Analysis
```
Per partition:
- Centroid: D floats (768 * 4 = 3KB for BERT-sized)
- N-gram dist: 256 floats (~1KB)
- Stats: ~20 floats (~80 bytes)

Total for K=256 partitions: ~1MB
Compare to full sphere (N=1M points): ~3GB
Compression ratio: 3000:1
```

## Research Questions

### 1. Energy Function over Raw Bytes

**Question**: How to define E over raw bytes (not just embeddings)?

**Options**:
- A. Use byte n-gram statistics as input features
- B. Learn a small MLP that maps bytes → energy
- C. Use existing BLT entropy as a prior: E_byte = -log P(byte | context)

**Recommendation**: Start with (C) since BLT entropy is already computed.

### 2. Tree Position Supervision

**Question**: Can tree position supervise the diffusion target?

**Hypothesis**: Bytes at similar tree positions (same parent branch) should have similar compressed representations.

**Proposed Loss**:
```
L_tree = ||compress(bytes_i) - compress(bytes_j)||^2 * (1 - tree_distance(i,j))
```
Where `tree_distance` is normalized path distance in hypergraph.

### 3. Connection to VQ-VAE / Discrete Diffusion

The ROOTS layer resembles a vector-quantized codebook:
- Each partition centroid is like a VQ codebook entry
- Routing to partition is like VQ assignment

**Discrete Diffusion Literature**:
- "Structured Denoising Diffusion Models" (Austin et al., 2021)
- "D3PM: Discrete Denoising Diffusion Probabilistic Models" (Hoogeboom et al., 2021)

**Key Difference**: Our approach uses continuous sphere coordinates with discrete hypergraph structure, rather than purely discrete diffusion.

## Implementation Path

### Phase 1: Static ROOTS (Done via max-cut)
Use existing `IsingEBM` for Max-Cut partitioning to create initial ROOTS clusters.

### Phase 2: Learned Routing
Train `PatchClassifierEBM` (plan section 7.3) to learn which partition to route to.

### Phase 3: Learned Compression (Future)
Extend the energy function to include byte-level terms:
```rust
pub struct ByteAwareEBM {
    // Existing embedding energy
    embedding_ebm: SphereEBM,
    
    // Byte-level energy (learned)
    byte_energy_weight: f32,
    
    // Tree coherence energy
    tree_coherence_weight: f32,
}

impl ByteAwareEBM {
    fn byte_energy(&self, bytes: &[u8], context: &ByteContext) -> f32 {
        // Use BLT entropy as base
        let blt_entropy = self.blt_model.entropy(bytes, context);
        
        // Add learned correction
        let correction = self.byte_mlp.forward(bytes);
        
        blt_entropy + correction
    }
}
```

## Score Matching Training

The score function is the gradient of log probability:
```
s(x) = ∇_x log p(x) = -∇_x E(x)
```

Training objective (denoising score matching):
```
L = E[||s(x_t) - ∇_x log p(x_t | x_0)||^2]
```

Since our Langevin sampler already computes `∇E`, we can:
1. Corrupt data with noise (forward pass)
2. Train energy function to predict noise direction (reverse)

## Next Steps

1. **Implement ROOTS partitioning** using existing `IsingEBM` + greedy coloring
2. **Create `ByteContext` struct** to hold n-gram statistics
3. **Design byte energy MLP** as extension to `SphereEBM`
4. **Benchmark compression** ratio vs retrieval accuracy

## References

- Extropic THRML: Original JAX implementation of thermodynamic machine learning
- BLT: "Better Language Representations via Byte-Level Tokenization"
- GraphMERT: Small graphical models outperforming large LLMs on structured retrieval
- Score matching: "Estimation of Non-Normalized Statistical Models"
- Discrete diffusion: "D3PM: Discrete Denoising Diffusion Probabilistic Models"

## Code Locations

- Langevin dynamics: `thrml-samplers/src/langevin.rs`
- Ising Max-Cut: `thrml-models/src/ising.rs`
- Navigator training: `thrml-sphere/src/training.rs`
- BLT entropy: `blt-burn/src/blt_core.rs`


