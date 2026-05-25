// Copyright © 2026 RNT56.

import Foundation

public func selectTurboQuantAttentionPath(
    request: TurboQuantAttentionRequest,
    capabilities: TurboQuantKernelCapabilities
) -> TurboQuantAttentionDecision {
    selectTurboQuantAttentionPath(
        request: request,
        capabilities: capabilities.attentionCapabilities
    )
}

public func selectTurboQuantAttentionPath(
    request: TurboQuantAttentionRequest,
    capabilities: TurboQuantAttentionCapabilities
) -> TurboQuantAttentionDecision {
    do {
        return try turboQuantAttentionDecision(request: request, capabilities: capabilities)
    } catch {
        var rejected = turboQuantRejectedPathsForUnavailableDecision(error)

        if request.fallbackState.packedFallbackAvailable {
            return TurboQuantAttentionDecision(
                selectedPath: .mlxPackedFallback,
                outputDType: request.outputDType,
                rejectedPaths: rejected
            )
        }

        if request.fallbackState.decodedFallbackAvailable || request.fallbackState.baselineAvailable {
            return TurboQuantAttentionDecision(
                selectedPath: .baseline,
                outputDType: request.outputDType,
                rejectedPaths: rejected
            )
        }

        rejected.append(
            RejectedPath(
                path: .mlxPackedFallback,
                reason: "packed fallback is unavailable or not allowed"
            )
        )
        rejected.append(
            RejectedPath(
                path: .baseline,
                reason: "baseline fallback is unavailable or not allowed"
            )
        )

        return TurboQuantAttentionDecision(
            selectedPath: .baseline,
            outputDType: request.outputDType,
            rejectedPaths: rejected
        )
    }
}

private func turboQuantRejectedPathsForUnavailableDecision(_ error: Error) -> [RejectedPath] {
    let reason = String(describing: error)
    let message = reason.isEmpty ? "path unavailable" : reason
    return [
        RejectedPath(path: .onlineFused, reason: message),
        RejectedPath(path: .tiledOnlineFused, reason: message),
        RejectedPath(path: .twoStageCompressed, reason: message),
    ]
}
