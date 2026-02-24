import Foundation

@MainActor
@Observable
class SystemMetricsVM {
    var cpuHistory = RollingBuffer(capacity: 900)      // 30 min at 2s
    var memHistory = RollingBuffer(capacity: 900)
    var diskReadHistory = RollingBuffer(capacity: 900)
    var netInHistory = RollingBuffer(capacity: 900)
    var gpuHistory = RollingBuffer(capacity: 900)

    var cpuForecast: [Float] = []
    var memForecast: [Float] = []
    var gpuForecast: [Float] = []

    var latestSnapshot = MetricsSnapshot()
    var alert: String?
    var isPredicting = false
    var isCollecting = false

    private let collector = MetricsCollector()
    private var collectTask: Task<Void, Never>?

    func startCollecting() {
        guard !isCollecting else { return }
        isCollecting = true
        // Prime the collector with a first sample (delta needs two readings)
        _ = collector.collect()

        collectTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(2))
                let snap = collector.collect()
                latestSnapshot = snap
                cpuHistory.append(snap.cpuPercent)
                memHistory.append(snap.memoryPercent)
                diskReadHistory.append(snap.diskReadMBps)
                netInHistory.append(snap.netInMBps)
                gpuHistory.append(snap.gpuPercent)
            }
        }
    }

    func stopCollecting() {
        collectTask?.cancel()
        collectTask = nil
        isCollecting = false
    }

    func predict(runner: TotoRunner) {
        guard runner.isModelLoaded else { return }
        isPredicting = true

        let predictionLength = 128  // ~4 min ahead at 2s intervals

        // Multivariate: CPU + Memory + GPU
        let vars = [
            cpuHistory.toArray(),
            memHistory.toArray(),
            gpuHistory.toArray(),
        ].filter { $0.count >= 64 }

        guard !vars.isEmpty else {
            isPredicting = false
            return
        }

        if let forecasts = runner.forecastMultivariate(
            variables: vars, predictionLength: predictionLength)
        {
            cpuForecast = forecasts[0].map { max(0, min(100, $0)) }
            if forecasts.count > 1 { memForecast = forecasts[1].map { max(0, min(100, $0)) } }
            if forecasts.count > 2 { gpuForecast = forecasts[2].map { max(0, min(100, $0)) } }

            // Check for alerts
            if let peakMem = memForecast.max(), peakMem > 90 {
                let minsToHigh =
                    memForecast.firstIndex(where: { $0 > 90 }).map { $0 * 2 / 60 } ?? 0
                alert = "Memory predicted to exceed 90% in ~\(max(1, minsToHigh)) min"
            } else if let peakCPU = cpuForecast.max(), peakCPU > 90 {
                alert = "CPU predicted to spike above 90%"
            } else {
                alert = nil
            }
        }

        isPredicting = false
    }
}
