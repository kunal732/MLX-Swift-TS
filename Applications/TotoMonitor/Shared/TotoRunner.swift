import Foundation
import MLX
import MLXTimeSeries

/// Shared Toto model loader and forecast runner.
@MainActor
@Observable
class TotoRunner {
    var isModelLoaded = false
    var isLoading = false
    var loadProgress: Double = 0
    var statusMessage = "Model not loaded"

    private var forecaster: TotoForecaster?

    func loadModel() async {
        guard !isModelLoaded && !isLoading else { return }
        isLoading = true
        loadProgress = 0
        statusMessage = "Downloading Toto (~605 MB)..."

        do {
            forecaster = try await TotoForecaster.loadFromHub(
                id: "kunal732/Toto-Open-Base-1.0-MLX"
            ) { [weak self] progress in
                Task { @MainActor [weak self] in
                    self?.loadProgress = progress.fractionCompleted
                    let mb = Int(progress.completedUnitCount / 1_000_000)
                    let total = max(1, Int(progress.totalUnitCount / 1_000_000))
                    self?.statusMessage = "Downloading... \(mb)/\(total) MB"
                }
            }
            isModelLoaded = true
            statusMessage = "Model ready"
        } catch {
            statusMessage = "Load failed: \(error.localizedDescription)"
        }
        isLoading = false
        loadProgress = 0
    }

    /// Run a univariate forecast on a single time series.
    func forecast(values: [Float], predictionLength: Int) -> [Float]? {
        guard let forecaster, values.count >= 64 else { return nil }
        let input = TimeSeriesInput.univariate(values)
        let result = forecaster.forecast(input: input, predictionLength: predictionLength)
        let arr = result.mean.squeezed()
        eval(arr)
        return (0 ..< predictionLength).map { arr[$0].item(Float.self) }
    }

    /// Run a multivariate forecast on multiple time series.
    func forecastMultivariate(variables: [[Float]], predictionLength: Int) -> [[Float]]? {
        guard let forecaster else { return nil }
        let V = variables.count
        guard V > 0 else { return nil }
        let T = variables[0].count
        guard T >= 64 else { return nil }

        var flat = [Float]()
        for v in 0 ..< V { flat.append(contentsOf: variables[v]) }

        let series = MLXArray(flat).reshaped(1, V, T)
        let mask = MLXArray.ones([1, V, T])
        let idMask = MLXArray.zeros([1, V], type: Int32.self)

        let input = TimeSeriesInput(
            series: series, paddingMask: mask, idMask: idMask
        ).padded(toPatchSize: 64)

        let result = forecaster.forecast(input: input, predictionLength: predictionLength)
        let forecastAll = result.mean.squeezed(axis: 0)  // [V, predictionLength]
        eval(forecastAll)

        return (0 ..< V).map { v in
            (0 ..< predictionLength).map { t in forecastAll[v, t].item(Float.self) }
        }
    }
}
