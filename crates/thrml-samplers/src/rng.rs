use rand::{RngCore, SeedableRng};
/// Simple RNG key system for deterministic random number generation.
///
/// This provides a functional-style RNG key system similar to JAX's key splitting.
/// Keys are represented as u64 seeds, and we use ChaCha8 for deterministic splitting.
use rand_chacha::ChaCha8Rng;

/// An RNG key for deterministic random number generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RngKey(pub u64);

impl RngKey {
    /// Create a new RNG key from a seed.
    pub fn new(seed: u64) -> Self {
        RngKey(seed)
    }

    /// Split this key into multiple independent keys.
    /// This is similar to JAX's `jax.random.split`.
    pub fn split(self, n: usize) -> Vec<RngKey> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![self];
        }

        // Use ChaCha8 to generate deterministic splits
        let mut rng = ChaCha8Rng::seed_from_u64(self.0);
        let mut keys = Vec::with_capacity(n);

        for _ in 0..n {
            let seed = rng.next_u64();
            keys.push(RngKey(seed));
        }

        keys
    }

    /// Split into exactly two keys (common case).
    pub fn split_two(self) -> (RngKey, RngKey) {
        let keys = self.split(2);
        (keys[0], keys[1])
    }

    /// Get the seed value.
    pub fn seed(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_key_split() {
        let key = RngKey::new(42);
        let keys = key.split(5);

        // Should get exactly 5 keys
        assert_eq!(keys.len(), 5);

        // All keys should be different
        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                assert_ne!(keys[i].0, keys[j].0, "Keys should be unique");
            }
        }
    }

    #[test]
    fn test_rng_key_deterministic() {
        let key1 = RngKey::new(42);
        let key2 = RngKey::new(42);

        let keys1 = key1.split(10);
        let keys2 = key2.split(10);

        // Same seed should produce same splits
        assert_eq!(keys1.len(), keys2.len());
        for (k1, k2) in keys1.iter().zip(keys2.iter()) {
            assert_eq!(k1.0, k2.0, "Same seed should produce same keys");
        }
    }
}
