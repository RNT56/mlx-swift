// Copyright © 2026 Schtack.

import Foundation
import MLX
import MLXNN

private struct ConverterArguments {
    var input: URL?
    var output: URL?
    var preset: TurboQuantPreset = .turbo4v2
    var groupSize = 64
    var mode: QuantizationMode = .affine
    var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    var valueBits: Int?
    var includePatterns = ["*.weight"]
    var excludePatterns = [
        "*.embed_tokens.weight",
        "*.embedding.weight",
        "*.lm_head.weight",
        "*.norm.weight",
    ]
    var overwrite = false
    var dryRun = false
}

@main
private enum TurboQuantConverter {
    static func main() throws {
        do {
            let arguments = try parseArguments(Array(CommandLine.arguments.dropFirst()))
            guard let input = arguments.input, let output = arguments.output else {
                printUsage()
                Foundation.exit(2)
            }

            let options = TurboQuantCheckpointConversionOptions(
                preset: arguments.preset,
                groupSize: arguments.groupSize,
                mode: arguments.mode,
                seed: arguments.seed,
                valueBits: arguments.valueBits,
                includePatterns: arguments.includePatterns,
                excludePatterns: arguments.excludePatterns,
                overwrite: arguments.overwrite
            )

            if arguments.dryRun {
                let reports = try dryRun(input: input, options: options)
                printReports(reports, output: output, dryRun: true)
                return
            }

            let reports = try turboQuantConvertCheckpoint(from: input, to: output, options: options)
            printReports(reports, output: output, dryRun: false)
        } catch {
            FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
            Foundation.exit(1)
        }
    }
}

private func parseArguments(_ raw: [String]) throws -> ConverterArguments {
    var result = ConverterArguments()
    var index = 0

    while index < raw.count {
        let arg = raw[index]
        switch arg {
        case "-h", "--help":
            printUsage()
            Foundation.exit(0)
        case "--input", "-i":
            result.input = URL(fileURLWithPath: try value(after: arg, raw, &index))
        case "--output", "-o":
            result.output = URL(fileURLWithPath: try value(after: arg, raw, &index))
        case "--preset":
            let rawPreset = try value(after: arg, raw, &index)
            guard let preset = TurboQuantPreset(rawValue: rawPreset) else {
                throw TurboQuantCheckpointConversionError.invalidConfiguration("unknown preset \(rawPreset)")
            }
            result.preset = preset
        case "--group-size":
            result.groupSize = try intValue(after: arg, raw, &index)
        case "--mode":
            let rawMode = try value(after: arg, raw, &index)
            guard let mode = QuantizationMode(rawValue: rawMode) else {
                throw TurboQuantCheckpointConversionError.invalidConfiguration("unknown mode \(rawMode)")
            }
            result.mode = mode
        case "--seed":
            result.seed = try uint64Value(after: arg, raw, &index)
        case "--value-bits":
            result.valueBits = try intValue(after: arg, raw, &index)
        case "--include":
            result.includePatterns.append(try value(after: arg, raw, &index))
        case "--exclude":
            result.excludePatterns.append(try value(after: arg, raw, &index))
        case "--only":
            result.includePatterns = [try value(after: arg, raw, &index)]
        case "--overwrite":
            result.overwrite = true
        case "--dry-run":
            result.dryRun = true
        default:
            if result.input == nil {
                result.input = URL(fileURLWithPath: arg)
            } else if result.output == nil {
                result.output = URL(fileURLWithPath: arg)
            } else {
                throw TurboQuantCheckpointConversionError.invalidConfiguration("unexpected argument \(arg)")
            }
        }
        index += 1
    }

    return result
}

private func value(after flag: String, _ raw: [String], _ index: inout Int) throws -> String {
    let next = index + 1
    guard next < raw.count else {
        throw TurboQuantCheckpointConversionError.invalidConfiguration("missing value for \(flag)")
    }
    index = next
    return raw[next]
}

private func intValue(after flag: String, _ raw: [String], _ index: inout Int) throws -> Int {
    let rawValue = try value(after: flag, raw, &index)
    guard let value = Int(rawValue) else {
        throw TurboQuantCheckpointConversionError.invalidConfiguration("invalid integer for \(flag): \(rawValue)")
    }
    return value
}

private func uint64Value(after flag: String, _ raw: [String], _ index: inout Int) throws -> UInt64 {
    let rawValue = try value(after: flag, raw, &index)
    let trimmed = rawValue.lowercased().replacingOccurrences(of: "_", with: "")
    if trimmed.hasPrefix("0x"), let value = UInt64(trimmed.dropFirst(2), radix: 16) {
        return value
    }
    guard let value = UInt64(trimmed) else {
        throw TurboQuantCheckpointConversionError.invalidConfiguration("invalid UInt64 for \(flag): \(rawValue)")
    }
    return value
}

private func dryRun(
    input: URL,
    options: TurboQuantCheckpointConversionOptions
) throws -> [TurboQuantCheckpointConversionReport] {
    let fileManager = FileManager.default
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: input.path(percentEncoded: false), isDirectory: &isDirectory) else {
        throw TurboQuantCheckpointConversionError.inputNotFound(input)
    }

    let files: [URL]
    if isDirectory.boolValue {
        files = try fileManager.contentsOfDirectory(at: input, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    } else {
        files = [input]
    }

    return try files.map { file in
        let (arrays, metadata) = try loadArraysAndMetadata(url: file)
        return try turboQuantConvertedArrays(arrays, metadata: metadata, options: options).report
    }
}

private func printReports(
    _ reports: [TurboQuantCheckpointConversionReport],
    output: URL,
    dryRun: Bool
) {
    let converted = reports.flatMap(\.converted)
    let skipped = reports.flatMap(\.skipped)
    let originalBytes = converted.reduce(0) { $0 + $1.originalBytes }
    let storedBytes = converted.reduce(0) { $0 + $1.storedBytes }
    let ratio = storedBytes > 0 ? Double(originalBytes) / Double(storedBytes) : 0

    print("\(dryRun ? "Would convert" : "Converted") \(converted.count) tensors")
    print("Skipped \(skipped.count) tensors")
    print(String(format: "Linear weight compression: %.2fx", ratio))
    print("Output: \(output.path(percentEncoded: false))")

    for tensor in converted.prefix(12) {
        print(
            String(
                format: "  %@ %@ -> %@ %.2fx",
                tensor.name,
                tensor.originalShape.description,
                tensor.packedShape.description,
                tensor.compressionRatio
            )
        )
    }
    if converted.count > 12 {
        print("  ... \(converted.count - 12) more")
    }
}

private func printUsage() {
    print(
        """
        Usage:
          swift run TurboQuantConverter --input <model-dir|weights.safetensors> --output <output-dir|out.safetensors> [options]

        Options:
          --preset <turbo4v2|turbo4|turbo3_5|turbo2_5>   Default: turbo4v2
          --group-size <int>                              Default: 64
          --mode <affine|mxfp4|mxfp8|nvfp4>               Default: affine
          --seed <uint64|0xhex>                            Default: 0x9E3779B97F4A7C15
          --value-bits <int>                               Value stream bit width metadata
          --include <glob>                                 Add tensor include glob
          --exclude <glob>                                 Add tensor exclude glob
          --only <glob>                                    Replace include globs with one glob
          --dry-run                                        Report without writing
          --overwrite                                      Replace existing output
        """
    )
}
