import Foundation
import PackagePlugin

@main
struct BuildSwiftPMMetalLibrary: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: any Target) async throws -> [Command] {
        #if os(Linux)
            return []
        #else
            let packageRoot = context.package.directoryURL
            let script = packageRoot
                .appendingPathComponent("tools")
                .appendingPathComponent("build-swiftpm-metallib.sh")
            let output = context.pluginWorkDirectoryURL.appendingPathComponent("default.metallib")

            return [
                .buildCommand(
                    displayName: "Build SwiftPM default.metallib",
                    executable: URL(fileURLWithPath: "/bin/bash"),
                    arguments: [script.path, output.path],
                    inputFiles: inputFiles(packageRoot: packageRoot, script: script),
                    outputFiles: [output]
                )
            ]
        #endif
    }

    #if !os(Linux)
        private func inputFiles(packageRoot: URL, script: URL) -> [URL] {
            let kernelsDirectory = [
                "Source", "Cmlx", "mlx", "mlx", "backend", "metal", "kernels",
            ].reduce(packageRoot) { url, component in
                url.appendingPathComponent(component)
            }
            var files = [script]
            files.append(contentsOf: recursivelyCollectedMetalInputs(in: kernelsDirectory))
            return files
        }

        private func recursivelyCollectedMetalInputs(in directory: URL) -> [URL] {
            let fileManager = FileManager.default
            guard let enumerator = fileManager.enumerator(atPath: directory.path) else {
                return []
            }

            return enumerator.compactMap { entry -> URL? in
                guard let entry = entry as? String else { return nil }
                guard entry.hasSuffix(".metal") || entry.hasSuffix(".h") else { return nil }
                return directory.appendingPathComponent(entry)
            }.sorted { $0.path < $1.path }
        }
    #endif
}
