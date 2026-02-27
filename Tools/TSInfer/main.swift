/// ts-infer: CLI tool for loading and testing MLX time series models.
///
/// Usage:
///   swift run TSInfer --hf-path kunal732/Toto-Open-Base-1.0-MLX
///   swift run TSInfer --mlx-path ./mlx_model
///   swift run TSInfer --hf-path kunal732/chronos-t5-base-mlx --prediction-length 16

import Foundation
import MLX
import MLXTimeSeries

// MARK: - Argument parsing

var hfPath: String? = nil
var mlxPath: String? = nil
var predictionLength: Int = 8

var args = CommandLine.arguments.dropFirst()
while !args.isEmpty {
    let arg = args.removeFirst()
    switch arg {
    case "--hf-path":
        hfPath = args.removeFirst()
    case "--mlx-path":
        mlxPath = args.removeFirst()
    case "--prediction-length":
        predictionLength = Int(args.removeFirst()) ?? 8
    default:
        break
    }
}

guard hfPath != nil || mlxPath != nil else {
    print("Usage: swift run TSInfer [--hf-path <HF_ID>] [--mlx-path <DIR>] [--prediction-length <N>]")
    print("Example: swift run TSInfer --hf-path kunal732/Toto-Open-Base-1.0-MLX")
    exit(1)
}

// MARK: - Run

// Synthetic input: 100 data points with a gentle sine wave pattern
let T = 100
let input: [Float] = (0..<T).map { i in
    let x = Float(i) / Float(T)
    return 50 + 20 * sin(2 * .pi * x * 3) + Float.random(in: -2...2)
}

print("Input: \(T) samples, range [\(input.min()!.rounded())...\(input.max()!.rounded())]")
print("Prediction length: \(predictionLength)")
print()

// warm up

Task {
    do {
        print("Loading model...")
        let forecaster: TimeSeriesForecaster
        if let id = hfPath {
            forecaster = try await TimeSeriesForecaster.loadFromHub(id: id) { progress in
                let pct = Int(progress.fractionCompleted * 100)
                let mb = Int(progress.completedUnitCount / 1_000_000)
                let total = max(1, Int(progress.totalUnitCount / 1_000_000))
                print("  Downloading... \(pct)% (\(mb)/\(total) MB)    ", terminator: "\r")
                fflush(stdout)
            }
            print("\nLoaded from Hub: \(id)")
        } else {
            let url = URL(filePath: mlxPath!)
            forecaster = try TimeSeriesForecaster.loadFromDirectory(url)
            print("Loaded from directory: \(mlxPath!)")
        }

        print("Model type: \(forecaster.modelType)")
        print()

        print("Running forecast...")
        let start = Date()
        let tsInput = TimeSeriesInput.univariate(input)
        let prediction = forecaster.forecast(input: tsInput, predictionLength: predictionLength)

        // Force evaluation (requires Metal GPU — runs fine when built via Xcode or
        // when the MLX metallib is available in the build).
        // On plain `swift run` without a bundled metallib this will crash; run via
        // Xcode or `swift build -c release && .build/release/TSInfer ...` instead.
        // Reshape to [predictionLength] regardless of batch/variate dims.
        let mean = prediction.mean.reshaped(-1)
        eval(mean)

        let elapsed = Date().timeIntervalSince(start) * 1000
        print("Inference: \(String(format: "%.0f", elapsed)) ms")
        print()

        let values = (0..<predictionLength).map { mean[$0].item(Float.self) }
        print("Forecast (next \(predictionLength) steps):")
        for (i, v) in values.enumerated() {
            print("  step \(String(format: "%2d", i+1)): \(String(format: "%.2f", v))")
        }

        if let q = prediction.quantiles {
            eval(q)
            print("\nQuantiles available: shape \(q.shape)")
        }

        print("\nOK - model loaded and inference succeeded")
        exit(0)

    } catch {
        print("ERROR: \(error)")
        print()
        if let desc = (error as? LocalizedError)?.errorDescription {
            print(desc)
        }
        exit(1)
    }
}

// Keep the main thread alive for the async task
RunLoop.main.run()
