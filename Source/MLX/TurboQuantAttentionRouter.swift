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
        let fallbackReason = rejected.map { "\($0.path.rawValue): \($0.reason)" }.joined(separator: "; ")

        func fallbackDecision(_ path: TurboQuantAttentionPath) -> TurboQuantAttentionDecision {
            TurboQuantAttentionDecision(
                selectedPath: path,
                outputDType: request.outputDType,
                rejectedPaths: rejected,
                headDimension: request.queryShape.indices.contains(3) ? request.queryShape[3] : nil,
                queryLength: request.queryShape.indices.contains(2) ? request.queryShape[2] : nil,
                logicalLength: request.keyLayout.logicalLength,
                dtype: "\(request.queryDType)->\(request.outputDType)",
                maskKind: request.maskKind.rawValue,
                kernelProfile: TurboQuantRuntimeProbe.shared.selectedKernelProfileWithoutRunningProbe(),
                fallbackReason: fallbackReason
            )
        }

        if request.fallbackState.packedFallbackAvailable {
            return fallbackDecision(.mlxPackedFallback)
        }

        if request.fallbackState.decodedFallbackAvailable || request.fallbackState.baselineAvailable {
            return fallbackDecision(.baseline)
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
            selectedPath: .unavailable,
            outputDType: request.outputDType,
            rejectedPaths: rejected,
            headDimension: request.queryShape.indices.contains(3) ? request.queryShape[3] : nil,
            queryLength: request.queryShape.indices.contains(2) ? request.queryShape[2] : nil,
            logicalLength: request.keyLayout.logicalLength,
            dtype: "\(request.queryDType)->\(request.outputDType)",
            maskKind: request.maskKind.rawValue,
            kernelProfile: TurboQuantRuntimeProbe.shared.selectedKernelProfileWithoutRunningProbe(),
            fallbackReason: rejected.map { "\($0.path.rawValue): \($0.reason)" }.joined(separator: "; ")
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
