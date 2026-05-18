# TurboQuant Linear Weights

Use ``TurboQuantLinear`` when a model needs TurboQuant-packed linear weights in
addition to TurboQuant KV-cache compression. The layer keeps an MLX-packed
fallback representation for portable execution and uses the verified
PolarQuant/QJL Metal path when the runtime supports it.

Convert an in-memory model:

```swift
turboQuantize(model: model, preset: .turbo4v2, groupSize: 64)
```

Convert safetensor checkpoint weights:

```sh
swift run TurboQuantConverter \
  --input /path/to/model \
  --output /path/to/model-turboquant \
  --preset turbo4v2 \
  --group-size 64
```

The converter replaces rank-2 `*.weight` tensors with packed weights and writes
matching `*.scales` and `*.biases` tensors. Embeddings, normalization weights,
and `lm_head` are excluded by default. The output safetensors include metadata
such as `quant_method=turboquant`, `linear_class=TurboQuantLinear`, preset, group
size, mode, seed, and value-bit settings so loaders can select
``TurboQuantLinear`` instead of ordinary ``QuantizedLinear``.

`--dry-run` reports the tensors that would be converted and the effective linear
weight compression ratio without writing files.
