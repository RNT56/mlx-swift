# Agent Notes for the Schtack MLX Swift Fork

This repository is the `RNT56/mlx-swift` fork used by Schtack projects. Treat it as an integrated fork, not a disposable experiment branch.

## Branch Model

- `main` is the current integrated fork state for downstream consumers.
- `schtack/turboquant-kv` intentionally points at the same commit as `main`. Keep this branch as a named integration branch for Schtack work.
- Do not assume older topic branches are active. Most remote branches are historical experiments or upstream maintenance work.
- Do not move `RNT56/mlx` `main` to Schtack-only patches. The MLX core patch is consumed by this repo via an explicit submodule pin.

As of 2026-05-16, active branches should contain this expected code baseline:

```text
dd13c2b55a743473d458058e9d9fb028233065ec
Point mlx submodule at Metal fallback fork
```

Branch heads may be later docs-only or maintenance commits, but they should not drop this baseline unless the fork stack is intentionally rebuilt.

## Dependency Stack

The intended stack is:

```text
RNT56/mlx-swift-lm
  -> RNT56/mlx-swift
      -> RNT56/mlx
```

This repo owns the `RNT56/mlx-swift -> RNT56/mlx` link through `Source/Cmlx/mlx`.

The expected MLX submodule pin is:

```text
Source/Cmlx/mlx:
f2ed827ef3c51ba7e5a0f7936fcb7c5cfcedb4e6
Add embedded Metal default library fallback
```

The expected `.gitmodules` entry for MLX is:

```text
url = https://github.com/RNT56/mlx
```

Do not change it back to `https://github.com/ml-explore/mlx` when preserving Schtack runtime behavior.

## Important Schtack Changes

The active branch must include the TurboQuant and robustness work, including:

- `24b3d0d` Add TurboQuant compressed attention kernels
- `e720346` Add v3 TurboQuant tiled rotating attention
- `bf234d5` Harden TurboQuant availability and shape contracts
- `a7641c6` Harden TurboQuant Metal template seeds
- `4993646` Add TurboQuant runtime capability probe
- `cf6d72f` Refine TurboQuant sustained profile selection
- `9c15aa4` Harden TurboQuant Metal runtime validation
- `b3cb256` Fix Metal fallback and linalg norm completeness
- `b269576` Add incomplete-marker audit and compile verification
- `dd13c2b` Point mlx submodule at Metal fallback fork

Before updating downstream pins, confirm the active branch contains these changes.

## Validation

Use focused checks before pushing:

```sh
git status --short --branch
git submodule status --recursive
swift package resolve
swift build --target MLX
swift build --target MLXNN
```

For changes touching TurboQuant or Metal runtime loading, also run the most relevant tests available locally.

## Downstream Pin Workflow

When this repo gets a new commit that downstream apps should consume:

1. Update `RNT56/mlx-swift-lm` `Package.swift` to pin the new `RNT56/mlx-swift` revision.
2. Validate and push `RNT56/mlx-swift-lm`.
3. Update downstream app pins, especially `RNT56/pines`, to the matching `mlx-swift` and `mlx-swift-lm` commits.

For `pines`, `project.yml` is the source of truth for XcodeGen package pins. Regenerate/check `Pines.xcodeproj/project.pbxproj` after changing it.
