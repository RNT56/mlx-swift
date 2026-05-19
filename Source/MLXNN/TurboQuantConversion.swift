// Copyright © 2026 RNT56.

import Foundation
import MLX

public enum TurboQuantCheckpointConversionError: Error, LocalizedError {
    case inputNotFound(URL)
    case unsupportedInput(URL)
    case outputExists(URL)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .inputNotFound(let url):
            "Input does not exist: \(url.path(percentEncoded: false))"
        case .unsupportedInput(let url):
            "Input must be a .safetensors file or a model directory: \(url.path(percentEncoded: false))"
        case .outputExists(let url):
            "Output already exists: \(url.path(percentEncoded: false))"
        case .invalidConfiguration(let message):
            message
        }
    }
}

public struct TurboQuantCheckpointConversionOptions: Sendable {
    public var preset: TurboQuantPreset
    public var groupSize: Int
    public var mode: QuantizationMode
    public var seed: UInt64
    public var valueBits: Int?
    public var includePatterns: [String]
    public var excludePatterns: [String]
    public var overwrite: Bool

    public init(
        preset: TurboQuantPreset = .turbo4v2,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        valueBits: Int? = nil,
        includePatterns: [String] = ["*.weight"],
        excludePatterns: [String] = [
            "*.embed_tokens.weight",
            "*.embedding.weight",
            "*.lm_head.weight",
            "*.norm.weight",
        ],
        overwrite: Bool = false
    ) {
        self.preset = preset
        self.groupSize = groupSize
        self.mode = mode
        self.seed = seed
        self.valueBits = valueBits
        self.includePatterns = includePatterns
        self.excludePatterns = excludePatterns
        self.overwrite = overwrite
    }
}

public struct TurboQuantConvertedTensor: Equatable, Sendable {
    public var name: String
    public var originalShape: [Int]
    public var packedShape: [Int]
    public var originalBytes: Int
    public var packedBytes: Int
    public var scaleBytes: Int
    public var biasBytes: Int

    public var storedBytes: Int {
        packedBytes + scaleBytes + biasBytes
    }

    public var compressionRatio: Double {
        guard storedBytes > 0 else { return 0 }
        return Double(originalBytes) / Double(storedBytes)
    }
}

public struct TurboQuantSkippedTensor: Equatable, Sendable {
    public var name: String
    public var reason: String
}

public struct TurboQuantCheckpointConversionReport: Equatable, Sendable {
    public var converted: [TurboQuantConvertedTensor]
    public var skipped: [TurboQuantSkippedTensor]
    public var metadata: [String: String]

    public var convertedCount: Int { converted.count }

    public var originalBytes: Int {
        converted.reduce(0) { $0 + $1.originalBytes }
    }

    public var storedBytes: Int {
        converted.reduce(0) { $0 + $1.storedBytes }
    }

    public var compressionRatio: Double {
        guard storedBytes > 0 else { return 0 }
        return Double(originalBytes) / Double(storedBytes)
    }
}

public func turboQuantConvertedArrays(
    _ arrays: [String: MLXArray],
    metadata: [String: String] = [:],
    options: TurboQuantCheckpointConversionOptions = TurboQuantCheckpointConversionOptions()
) throws -> (
    arrays: [String: MLXArray],
    metadata: [String: String],
    report: TurboQuantCheckpointConversionReport
) {
    try validateTurboQuantConversionOptions(options)

    var output = arrays
    var converted = [TurboQuantConvertedTensor]()
    var skipped = [TurboQuantSkippedTensor]()

    let configuration = TurboQuantConfiguration(
        preset: options.preset,
        role: .vector,
        groupSize: options.groupSize,
        mode: options.mode,
        backend: .mlxPacked,
        seed: options.seed,
        valueBits: options.valueBits ?? options.preset.defaultValueBits
    )

    for name in arrays.keys.sorted() {
        guard let array = arrays[name] else { continue }
        guard shouldConsiderTurboQuantTensor(name, options: options) else {
            continue
        }
        guard array.ndim == 2 else {
            skipped.append(
                .init(name: name, reason: "expected rank-2 linear weight, found \(array.shape)"))
            continue
        }
        guard array.dtype.isFloatingPoint, !array.dtype.isComplex else {
            skipped.append(
                .init(name: name, reason: "expected floating-point weight, found \(array.dtype)"))
            continue
        }
        let shape = array.shape
        guard shape.count == 2, shape[1] > 0, shape[1] % options.groupSize == 0 else {
            skipped.append(
                .init(
                    name: name,
                    reason: "input dimension must be divisible by group size \(options.groupSize)"))
            continue
        }

        let packed = turboQuantized(array, configuration: configuration)
        eval(packed.weight, packed.scales)
        if let biases = packed.biases {
            eval(biases)
        }

        output[name] = packed.weight
        let prefix = String(name.dropLast(".weight".count))
        output["\(prefix).scales"] = packed.scales
        if let biases = packed.biases {
            output["\(prefix).biases"] = biases
        } else {
            output.removeValue(forKey: "\(prefix).biases")
        }

        converted.append(
            TurboQuantConvertedTensor(
                name: name,
                originalShape: shape,
                packedShape: packed.weight.shape,
                originalBytes: array.nbytes,
                packedBytes: packed.weight.nbytes,
                scaleBytes: packed.scales.nbytes,
                biasBytes: packed.biases?.nbytes ?? 0
            )
        )
    }

    let conversionMetadata = turboQuantConversionMetadata(
        base: metadata,
        options: options,
        convertedCount: converted.count
    )
    let report = TurboQuantCheckpointConversionReport(
        converted: converted,
        skipped: skipped,
        metadata: conversionMetadata
    )
    return (output, conversionMetadata, report)
}

public func turboQuantConvertSafetensors(
    from input: URL,
    to output: URL,
    options: TurboQuantCheckpointConversionOptions = TurboQuantCheckpointConversionOptions()
) throws -> TurboQuantCheckpointConversionReport {
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: input.path(percentEncoded: false)) else {
        throw TurboQuantCheckpointConversionError.inputNotFound(input)
    }
    guard options.overwrite || !fileManager.fileExists(atPath: output.path(percentEncoded: false))
    else {
        throw TurboQuantCheckpointConversionError.outputExists(output)
    }
    try fileManager.createDirectory(
        at: output.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )

    let (arrays, metadata) = try loadArraysAndMetadata(url: input)
    let converted = try turboQuantConvertedArrays(arrays, metadata: metadata, options: options)
    try save(arrays: converted.arrays, metadata: converted.metadata, url: output)
    return converted.report
}

public func turboQuantConvertCheckpoint(
    from input: URL,
    to output: URL,
    options: TurboQuantCheckpointConversionOptions = TurboQuantCheckpointConversionOptions()
) throws -> [TurboQuantCheckpointConversionReport] {
    let fileManager = FileManager.default
    let inputPath = input.path(percentEncoded: false)
    guard fileManager.fileExists(atPath: inputPath) else {
        throw TurboQuantCheckpointConversionError.inputNotFound(input)
    }

    var isDirectory: ObjCBool = false
    fileManager.fileExists(atPath: inputPath, isDirectory: &isDirectory)

    if !isDirectory.boolValue {
        guard input.pathExtension == "safetensors" else {
            throw TurboQuantCheckpointConversionError.unsupportedInput(input)
        }
        return [try turboQuantConvertSafetensors(from: input, to: output, options: options)]
    }

    guard options.overwrite || !fileManager.fileExists(atPath: output.path(percentEncoded: false))
    else {
        throw TurboQuantCheckpointConversionError.outputExists(output)
    }
    try fileManager.createDirectory(at: output, withIntermediateDirectories: true)

    let safetensorFiles = try fileManager.contentsOfDirectory(
        at: input,
        includingPropertiesForKeys: nil
    )
    .filter { $0.pathExtension == "safetensors" }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }

    guard !safetensorFiles.isEmpty else {
        throw TurboQuantCheckpointConversionError.unsupportedInput(input)
    }

    try copyNonSafetensorFiles(from: input, to: output, overwrite: options.overwrite)

    var reports = [TurboQuantCheckpointConversionReport]()
    for source in safetensorFiles {
        let destination = output.appendingPathComponent(source.lastPathComponent)
        reports.append(
            try turboQuantConvertSafetensors(from: source, to: destination, options: options))
    }
    try updateTurboQuantConfigJSON(in: output, options: options)
    return reports
}

private func validateTurboQuantConversionOptions(_ options: TurboQuantCheckpointConversionOptions)
    throws
{
    guard options.groupSize > 0 else {
        throw TurboQuantCheckpointConversionError.invalidConfiguration(
            "group size must be positive")
    }
    guard options.valueBits == nil || (options.valueBits! >= 1 && options.valueBits! <= 8) else {
        throw TurboQuantCheckpointConversionError.invalidConfiguration(
            "value bits must be between 1 and 8")
    }
}

private func shouldConsiderTurboQuantTensor(
    _ name: String,
    options: TurboQuantCheckpointConversionOptions
) -> Bool {
    options.includePatterns.contains { wildcardMatch(name, pattern: $0) }
        && !options.excludePatterns.contains { wildcardMatch(name, pattern: $0) }
}

private func turboQuantConversionMetadata(
    base metadata: [String: String],
    options: TurboQuantCheckpointConversionOptions,
    convertedCount: Int
) -> [String: String] {
    var result = metadata
    result["quant_method"] = "turboquant"
    result["linear_class"] = "TurboQuantLinear"
    result["turboquant_format"] = "mlx_packed"
    result["turboquant_preset"] = options.preset.rawValue
    result["turboquant_group_size"] = "\(options.groupSize)"
    result["turboquant_bits"] = "\(options.preset.effectiveBits)"
    result["turboquant_mode"] = options.mode.rawValue
    result["turboquant_seed"] = "\(options.seed)"
    result["turboquant_value_bits"] = "\(options.valueBits ?? options.preset.defaultValueBits)"
    result["turboquant_converted_tensors"] = "\(convertedCount)"
    return result
}

private func wildcardMatch(_ value: String, pattern: String) -> Bool {
    let escaped = NSRegularExpression.escapedPattern(for: pattern)
        .replacingOccurrences(of: "\\*", with: ".*")
        .replacingOccurrences(of: "\\?", with: ".")
    let regex = "^\(escaped)$"
    return value.range(of: regex, options: [.regularExpression]) != nil
}

private func copyNonSafetensorFiles(from input: URL, to output: URL, overwrite: Bool) throws {
    let fileManager = FileManager.default
    let files = try fileManager.contentsOfDirectory(
        at: input,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles]
    )
    for source in files where source.pathExtension != "safetensors" {
        let destination = output.appendingPathComponent(source.lastPathComponent)
        if overwrite, fileManager.fileExists(atPath: destination.path(percentEncoded: false)) {
            try fileManager.removeItem(at: destination)
        }
        if !fileManager.fileExists(atPath: destination.path(percentEncoded: false)) {
            try fileManager.copyItem(at: source, to: destination)
        }
    }
}

private func updateTurboQuantConfigJSON(
    in directory: URL,
    options: TurboQuantCheckpointConversionOptions
) throws {
    let configURL = directory.appendingPathComponent("config.json")
    guard FileManager.default.fileExists(atPath: configURL.path(percentEncoded: false)) else {
        return
    }
    let data = try Data(contentsOf: configURL)
    var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    object["quantization"] = [
        "quant_method": "turboquant",
        "linear_class": "TurboQuantLinear",
        "turboquant_format": "mlx_packed",
        "preset": options.preset.rawValue,
        "group_size": options.groupSize,
        "bits": options.preset.effectiveBits,
        "mode": options.mode.rawValue,
        "seed": String(options.seed),
        "value_bits": options.valueBits ?? options.preset.defaultValueBits,
    ]
    let encoded = try JSONSerialization.data(
        withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    try encoded.write(to: configURL)
}
