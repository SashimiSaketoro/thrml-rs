# thrml-observers

Observation utilities for the THRML probabilistic computing library.

## Overview

Observers allow you to collect data during sampling without modifying the sampling algorithm.

### StateObserver

Collect state samples for specific blocks:

```rust
use thrml_observers::{StateObserver, AbstractObserver};

let observer = StateObserver::new(blocks_to_observe);
let (carry, samples) = observer.observe(spec, free_state, clamped_state, carry, iteration, &device);
```

### MomentAccumulatorObserver

Compute running moment statistics:

```rust
use thrml_observers::moment::{MomentAccumulatorObserver, MomentSpec};

// Create observer for first and second moments
let spec = MomentAccumulatorObserver::ising_moment_spec(&nodes, &edges);
let observer = MomentAccumulatorObserver::new(spec, true); // true = spin transform
```

## AbstractObserver Trait

All observers implement:

```rust
pub trait AbstractObserver {
    type ObserveCarry: Clone;
    
    fn observe(
        &self,
        spec: &BlockSpec,
        state_free: &[Tensor],
        state_clamped: &[Tensor],
        carry: Self::ObserveCarry,
        iteration: usize,
        device: &WgpuDevice,
    ) -> (Self::ObserveCarry, Option<Vec<Tensor>>);
    
    fn init(&self, device: &WgpuDevice) -> Self::ObserveCarry;
}
```

## License

MIT OR Apache-2.0

