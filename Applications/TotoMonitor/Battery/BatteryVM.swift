import Foundation

@MainActor
@Observable
class BatteryVM {
    var percentHistory = RollingBuffer(capacity: 360)   // 1 hour at 10s
    var wattsHistory = RollingBuffer(capacity: 360)

    var percentForecast: [Float] = []
    var wattsForecast: [Float] = []

    var latestSnapshot: BatterySnapshot?
    var predictedTimeToEmpty: String?
    var isPredicting = false
    var isCollecting = false

    private let collector = BatteryCollector()
    private var collectTask: Task<Void, Never>?

    func startCollecting() {
        guard !isCollecting else { return }
        isCollecting = true

        collectTask = Task {
            while !Task.isCancelled {
                if let snap = collector.collect() {
                    latestSnapshot = snap
                    percentHistory.append(snap.percent)
                    wattsHistory.append(snap.watts)
                }
                try? await Task.sleep(for: .seconds(10))
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

        // Predict 1 hour ahead (360 samples at 10s = 1hr)
        let predLength = 360
        let vars = [
            percentHistory.toArray(),
            wattsHistory.toArray(),
        ].filter { $0.count >= 64 }

        guard !vars.isEmpty else {
            isPredicting = false
            return
        }

        if let forecasts = runner.forecastMultivariate(
            variables: vars, predictionLength: min(predLength, 128))
        {
            percentForecast = forecasts[0].map { max(0, min(100, $0)) }
            if forecasts.count > 1 { wattsForecast = forecasts[1].map { max(0, $0) } }

            // Estimate time to empty from predicted battery curve
            if let current = percentForecast.first, current > 0 {
                if let emptyIdx = percentForecast.firstIndex(where: { $0 <= 5 }) {
                    let minutes = emptyIdx * 10 / 60
                    predictedTimeToEmpty = "\(max(1, minutes)) min"
                } else if let last = percentForecast.last, let first = percentForecast.first,
                    first > last
                {
                    let ratePerSample = (first - last) / Float(percentForecast.count)
                    if ratePerSample > 0 {
                        let samplesToZero = first / ratePerSample
                        let minutes = Int(samplesToZero) * 10 / 60
                        predictedTimeToEmpty = "~\(minutes) min (extrapolated)"
                    }
                }
            }
        }

        isPredicting = false
    }
}
