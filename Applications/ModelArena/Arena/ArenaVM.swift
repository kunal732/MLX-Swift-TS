import Foundation

/// On-disk snapshot of collected metrics.
private struct MetricsCache: Codable {
    let cpu: [Float]
    let mem: [Float]
    let gpu: [Float]
    let thermal: [Float]?
    let battery: [Float]?
    let savedAt: Date
}

private let cacheURL: URL = {
    let dir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        .appending(component: "com.mlxtoto.ModelArena")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir.appending(component: "metrics_cache.json")
}()

/// Maximum age of cached samples before discarding (10 minutes).
private let maxCacheAge: TimeInterval = 600

@MainActor
@Observable
class ArenaVM {
    var cpuHistory = RollingBuffer(capacity: 900)      // 30 min at 2s
    var memHistory = RollingBuffer(capacity: 900)
    var gpuHistory = RollingBuffer(capacity: 900)

    #if os(iOS)
    var thermalHistory = RollingBuffer(capacity: 900)
    var batteryHistory = RollingBuffer(capacity: 900)
    #endif

    var latestSnapshot = MetricsSnapshot()
    var isCollecting = false
    var isPredicting = false

    /// Increments on every sample, used to drive chart re-renders.
    var tick = 0

    /// Previous forecasts keyed by model name, for live MASE scoring.
    /// Each entry stores (forecastValues, historySizeAtPredictionTime).
    var previousForecasts: [(modelName: String, forecast: [Float], startIndex: Int)] = []

    /// Tick value when the last forecast was made, used to slide the forecast window.
    var forecastTick = 0

    private let collector = MetricsCollector()
    private var collectTask: Task<Void, Never>?

    /// Reference to the runner for auto-prediction.
    var autoRunner: ArenaRunner?

    /// Auto-predict interval in collection ticks (10 ticks × 2s = 20s).
    private let autoPredictInterval = 10

    var sampleCount: Int { cpuHistory.count }
    var hasEnoughSamples: Bool { sampleCount >= 64 }

    init() {
        restoreFromDisk()
    }

    func startCollecting() {
        guard !isCollecting else { return }
        isCollecting = true
        _ = collector.collect()

        collectTask = Task {
            var ticksSinceSave = 0
            var ticksSincePredict = 0
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(2))
                let snap = collector.collect()
                latestSnapshot = snap
                cpuHistory.append(snap.cpuPercent)
                memHistory.append(snap.memoryPercent)
                gpuHistory.append(snap.gpuPercent)

                #if os(iOS)
                thermalHistory.append(snap.thermalState)
                batteryHistory.append(snap.batteryLevel)
                #endif

                tick += 1

                // Update MASE scores with new ground truth every tick
                if let runner = autoRunner, !previousForecasts.isEmpty {
                    updateMASE(runner: runner)
                }

                // Auto-save every 30 seconds
                ticksSinceSave += 1
                if ticksSinceSave >= 15 {
                    ticksSinceSave = 0
                    saveToDisk()
                }

                // Auto-predict every 20 seconds if models are loaded
                ticksSincePredict += 1
                if ticksSincePredict >= autoPredictInterval,
                   let runner = autoRunner,
                   !runner.slots.isEmpty,
                   hasEnoughSamples,
                   !isPredicting
                {
                    ticksSincePredict = 0
                    predict(runner: runner)
                }
            }
        }
    }

    func stopCollecting() {
        collectTask?.cancel()
        collectTask = nil
        isCollecting = false
        saveToDisk()
    }

    // MARK: - Persistence

    private func saveToDisk() {
        #if os(iOS)
        let cache = MetricsCache(
            cpu: cpuHistory.toArray(),
            mem: memHistory.toArray(),
            gpu: gpuHistory.toArray(),
            thermal: thermalHistory.toArray(),
            battery: batteryHistory.toArray(),
            savedAt: Date()
        )
        #else
        let cache = MetricsCache(
            cpu: cpuHistory.toArray(),
            mem: memHistory.toArray(),
            gpu: gpuHistory.toArray(),
            thermal: nil,
            battery: nil,
            savedAt: Date()
        )
        #endif
        if let data = try? JSONEncoder().encode(cache) {
            try? data.write(to: cacheURL, options: .atomic)
        }
    }

    private func restoreFromDisk() {
        guard let data = try? Data(contentsOf: cacheURL),
              let cache = try? JSONDecoder().decode(MetricsCache.self, from: data)
        else { return }

        // Discard if too old
        guard abs(cache.savedAt.timeIntervalSinceNow) < maxCacheAge else {
            try? FileManager.default.removeItem(at: cacheURL)
            return
        }

        cpuHistory = RollingBuffer(capacity: 900, restoring: cache.cpu)
        memHistory = RollingBuffer(capacity: 900, restoring: cache.mem)
        gpuHistory = RollingBuffer(capacity: 900, restoring: cache.gpu)

        #if os(iOS)
        if let thermal = cache.thermal {
            thermalHistory = RollingBuffer(capacity: 900, restoring: thermal)
        }
        if let battery = cache.battery {
            batteryHistory = RollingBuffer(capacity: 900, restoring: battery)
        }
        #endif

        print("ArenaVM: restored \(cache.cpu.count) samples from cache")
    }

    // MARK: - Prediction

    /// Run all models and update their slots with forecasts + timing.
    func predict(runner: ArenaRunner) {
        guard !runner.slots.isEmpty, hasEnoughSamples else { return }
        isPredicting = true

        let cpuValues = cpuHistory.toArray()
        let predictionLength = 128

        // Store reference point for MASE scoring
        let startIndex = cpuValues.count

        let runs = runner.forecastAll(values: cpuValues, predictionLength: predictionLength)
        forecastTick = tick

        // Save forecasts for live MASE computation
        previousForecasts = runs.map { run in
            (modelName: run.modelName, forecast: run.forecast, startIndex: startIndex)
        }

        // Update live MASE scores now (against whatever ground truth is available)
        updateMASE(runner: runner)

        isPredicting = false
    }

    /// Compute MASE for each model using ground truth that has arrived since the forecast.
    /// MASE = mean(|actual - forecast|) / mean(|actual[t] - actual[t-1]|)
    func updateMASE(runner: ArenaRunner) {
        let elapsed = forecastTick > 0 ? max(0, tick - forecastTick) : 0
        guard elapsed >= 2 else { return }

        let currentHistory = cpuHistory.toArray()
        guard currentHistory.count >= elapsed else { return }

        // Ground truth = the last `elapsed` samples (arrived since forecast)
        let groundTruth = Array(currentHistory.suffix(elapsed))
        // Naive baseline = the `elapsed` samples just before the forecast
        let preHistory = Array(currentHistory.dropLast(elapsed))

        for i in runner.slots.indices {
            guard let stored = previousForecasts.first(where: {
                $0.modelName == runner.slots[i].name
            }) else {
                runner.slots[i].liveMASE = nil
                continue
            }

            let steps = min(elapsed, stored.forecast.count)
            guard steps >= 2 else {
                runner.slots[i].liveMASE = nil
                continue
            }

            // Forecast error: mean(|actual - forecast|)
            var forecastError: Float = 0
            for t in 0..<steps {
                forecastError += abs(groundTruth[t] - stored.forecast[t])
            }
            forecastError /= Float(steps)

            // Naive error: mean(|actual[t] - actual[t-1]|) from pre-forecast history
            let naiveCount = min(steps, preHistory.count)
            guard naiveCount >= 2 else {
                runner.slots[i].liveMASE = nil
                continue
            }
            let naiveSlice = Array(preHistory.suffix(naiveCount))
            var naiveError: Float = 0
            for t in 1..<naiveSlice.count {
                naiveError += abs(naiveSlice[t] - naiveSlice[t - 1])
            }
            naiveError /= Float(naiveSlice.count - 1)

            if naiveError > 0.001 {
                runner.slots[i].liveMASE = Double(forecastError / naiveError)
            } else {
                runner.slots[i].liveMASE = Double(forecastError)
            }
        }
    }
}
