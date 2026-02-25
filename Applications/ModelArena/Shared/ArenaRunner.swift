import Foundation
import MLX
import MLXTimeSeries

/// MLX requires real Metal GPU — iOS Simulator has Metal but lacks MLX's shader kernels.
private var canRunMLX: Bool {
    #if targetEnvironment(simulator)
    return false
    #else
    return true
    #endif
}

/// A HuggingFace model to load into the arena.
struct HubModel: Identifiable, Hashable {
    let id: String
    let name: String
    let hubID: String

    init(name: String, hubID: String) {
        self.id = hubID
        self.name = name
        self.hubID = hubID
    }
}

/// Default models available in the arena.
let defaultHubModels: [HubModel] = [
    HubModel(name: "Toto Base", hubID: "kunal732/Toto-Open-Base-1.0-MLX"),
    HubModel(name: "TimesFM 2.5", hubID: "kunal732/timesfm-2.5-200m-transformers-mlx"),
]

/// A single model slot in the arena.
struct ModelSlot: Identifiable {
    let id = UUID()
    let name: String
    let modelType: String
    var forecaster: TimeSeriesForecaster
    var isLoaded: Bool = true
    var forecast: [Float] = []
    var inferenceMs: Double = 0
    var memoryMB: Double = 0
    var liveMASE: Double?
}

/// Result from a single model's forecast run.
struct ForecastRun {
    let modelName: String
    let forecast: [Float]
    let inferenceMs: Double
}

/// Multi-model loader and parallel forecaster for the arena.
@MainActor
@Observable
class ArenaRunner {
    var slots: [ModelSlot] = []
    var isLoading = false
    var loadingStatus = ""
    var loadProgress: Double = 0
    var currentModelIndex = 0
    var totalModelsToLoad = 0

    var loadedCount: Int { slots.filter(\.isLoaded).count }

    /// Scan a local directory for model subdirectories containing config.json.
    func loadFromDirectory(_ directory: URL) {
        guard canRunMLX else {
            loadingStatus = "No Metal GPU (simulator). Use a physical device."
            return
        }
        isLoading = true
        loadingStatus = "Scanning directory..."
        slots.removeAll()

        let fm = FileManager.default

        // Collect directories that contain config.json
        var modelDirs: [URL] = []
        if let contents = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.isDirectoryKey])
        {
            modelDirs = contents.filter { url in
                let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                let hasConfig = fm.fileExists(atPath: url.appending(component: "config.json").path)
                return isDir && hasConfig
            }
        }

        // Fallback: the directory itself might be a single model
        if modelDirs.isEmpty && fm.fileExists(atPath: directory.appending(component: "config.json").path) {
            modelDirs = [directory]
        }

        guard !modelDirs.isEmpty else {
            loadingStatus = "No models found in directory"
            isLoading = false
            return
        }

        var failures = [String]()
        for dir in modelDirs {
            let name = dir.lastPathComponent
            loadingStatus = "Loading \(name)..."
            do {
                let forecaster = try TimeSeriesForecaster.loadFromDirectory(dir)
                slots.append(ModelSlot(
                    name: name,
                    modelType: forecaster.modelType,
                    forecaster: forecaster
                ))
                print("ArenaRunner: loaded \(name) (\(forecaster.modelType))")
            } catch {
                print("ArenaRunner: FAILED \(name): \(error)")
                failures.append(name)
            }
        }

        if failures.isEmpty {
            loadingStatus = "\(slots.count) model(s) loaded"
        } else {
            loadingStatus = "\(slots.count) loaded, \(failures.count) failed (\(failures.joined(separator: ", ")))"
        }
        isLoading = false
    }

    /// Load models from HuggingFace Hub by their repo IDs.
    func loadFromHub(models: [HubModel]) async {
        guard canRunMLX else {
            loadingStatus = "No Metal GPU (simulator). Use a physical device."
            return
        }
        isLoading = true
        slots.removeAll()
        currentModelIndex = 0
        totalModelsToLoad = models.count

        for (idx, model) in models.enumerated() {
            currentModelIndex = idx + 1
            loadProgress = 0
            loadingStatus = "Downloading \(model.name) (\(currentModelIndex)/\(totalModelsToLoad))..."

            do {
                let forecaster = try await TimeSeriesForecaster.loadFromHub(
                    id: model.hubID
                ) { [weak self] progress in
                    Task { @MainActor [weak self] in
                        self?.loadProgress = progress.fractionCompleted
                        let mb = Int(progress.completedUnitCount / 1_000_000)
                        let total = max(1, Int(progress.totalUnitCount / 1_000_000))
                        self?.loadingStatus =
                            "Downloading \(model.name)... \(mb)/\(total) MB"
                    }
                }

                let slot = ModelSlot(
                    name: model.name,
                    modelType: forecaster.modelType,
                    forecaster: forecaster
                )
                slots.append(slot)
            } catch {
                loadingStatus = "Failed \(model.name): \(error.localizedDescription)"
                // Continue loading remaining models
                try? await Task.sleep(for: .seconds(1))
            }
        }

        loadingStatus = "\(slots.count) model(s) loaded"
        loadProgress = 0
        isLoading = false
    }

    /// Load a single model from HuggingFace Hub (for adding custom models).
    func addFromHub(id: String, name: String) async {
        guard canRunMLX else {
            loadingStatus = "No Metal GPU (simulator). Use a physical device."
            return
        }
        isLoading = true
        loadProgress = 0
        loadingStatus = "Downloading \(name)..."

        do {
            let forecaster = try await TimeSeriesForecaster.loadFromHub(
                id: id
            ) { [weak self] progress in
                Task { @MainActor [weak self] in
                    self?.loadProgress = progress.fractionCompleted
                    let mb = Int(progress.completedUnitCount / 1_000_000)
                    let total = max(1, Int(progress.totalUnitCount / 1_000_000))
                    self?.loadingStatus = "Downloading \(name)... \(mb)/\(total) MB"
                }
            }

            let slot = ModelSlot(
                name: name,
                modelType: forecaster.modelType,
                forecaster: forecaster
            )
            slots.append(slot)
            loadingStatus = "\(slots.count) model(s) loaded"
        } catch {
            loadingStatus = "Failed: \(error.localizedDescription)"
        }

        loadProgress = 0
        isLoading = false
    }

    /// Run forecast on all loaded models and return per-model results.
    func forecastAll(values: [Float], predictionLength: Int) -> [ForecastRun] {
        guard !slots.isEmpty, values.count >= 64 else { return [] }

        let input = TimeSeriesInput.univariate(values)
        var runs = [ForecastRun]()

        for i in slots.indices {
            let slot = slots[i]
            guard slot.isLoaded else { continue }

            let start = CFAbsoluteTimeGetCurrent()
            let prediction = slot.forecaster.forecast(
                input: input, predictionLength: predictionLength)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000

            let mean = prediction.mean.squeezed()
            eval(mean)
            let forecastValues = (0..<predictionLength).map { mean[$0].item(Float.self) }

            slots[i].forecast = forecastValues.map { max(0, min(100, $0)) }
            slots[i].inferenceMs = elapsed

            runs.append(ForecastRun(
                modelName: slot.name,
                forecast: slots[i].forecast,
                inferenceMs: elapsed
            ))
        }

        return runs
    }
}
