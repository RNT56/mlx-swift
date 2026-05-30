# Layout V5 and Kernels

W13 is the optimization lane. It must not activate before benchmark JSON, quality gates, and evidence import exist. Implementation may land behind a disabled flag, but product activation waits for measurement.

Launch wave: Wave 5. Implementation may exist behind a disabled flag earlier only if it does not disturb W1-W3 contracts or benchmark comparability.

## Worker

| Worker | Branch | Phase | Priority |
| --- | --- | --- | --- |
| W13 | `tq/layout-v5-kernels` | MVP 4 | P2 |

## Owned files

Likely areas:

- `Source/MLX/TurboQuant.swift`;
- `Source/MLX/TurboQuantLayoutV5.swift`;
- kernel templates/source strings;
- benchmark tests;
- quality A/B reports.

Do not modify admission or Pines evidence code from this repo.

## Feature flag

Layout V5 was implemented behind a flag and is now the default write layout for
real-device evidence collection:

```text
turboQuantLayoutV5 = on by default
```

V4 compatibility is required.

## Goals

- reduce actual bits/value or improve decode speed;
- preserve quality gates;
- avoid hidden full-cache copies;
- specialize fused decode kernels for common head dimensions;
- keep bfloat output separately gated.

## Layout V5 tasks

1. Add Layout V5 current/next version contract.
2. Add deterministic high-precision mask option.
3. Replace prior-coordinate loops with popcount offsets:

```swift
let highBefore = popcount(highMask & ((1 << local) - 1))
let bitOffset = local * baseBits + highBefore * (highBits - baseBits)
```

4. Add optional fp16 scale path if quality and speed support it.
5. Add V4 read compatibility.
6. Add migration tests.
7. Report actual bits/value.

## Fused specialization targets

Specialize supported decode paths for:

- 64;
- 80;
- 96;
- 128;
- 192;
- 256.

If current code already supports additional dimensions such as 112 or 240, keep them gated and benchmarked separately.

## Numeric requirements

- QK accumulation stays float.
- Online softmax max/sum stay float.
- Value decode uses token/dim tiles.
- bfloat output is independently gated.
- Unsupported head dimensions fall back safely.

## Hidden-copy requirements

- long compressed K/V inputs must not be copied wholesale before kernel launch;
- canonical layout validation should catch noncanonical storage;
- shape/stride indexing may be used where copy avoidance requires it;
- query input may copy if small and bounded;
- benchmark report should record hidden-copy audit status where possible.

## Quality requirements

Layout V5 cannot be product-enabled unless:

- QualityGate passes;
- fallback equivalence remains true;
- no NaN/Inf;
- prefill exactness remains true;
- top-1/KL/max-error thresholds remain inside profile gates.

## Acceptance

- V4 loads for compatibility comparisons.
- V5 is the default write layout.
- V5 improves speed or actual bits/value in benchmark.
- Fused path beats two-stage for Q=1 on supported dimensions.
- Unsupported dimensions fall back with typed reason.
- Benchmark JSON includes before/after.
