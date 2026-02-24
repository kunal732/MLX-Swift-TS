import Foundation

@MainActor
@Observable
class AppProfilerVM {
    var processes: [ProcessInfo2] = []
    var selectedPID: pid_t?
    var selectedName: String = ""

    var cpuHistory = RollingBuffer(capacity: 900)
    var memHistory = RollingBuffer(capacity: 900)

    var cpuForecast: [Float] = []
    var memForecast: [Float] = []

    var isPredicting = false
    var isCollecting = false

    private let collector = ProcessListCollector()
    private var collectTask: Task<Void, Never>?

    func startCollecting() {
        guard !isCollecting else { return }
        isCollecting = true

        collectTask = Task {
            while !Task.isCancelled {
                let all = collector.collectAll()
                processes = Array(all.prefix(50))  // top 50

                // Track selected process
                if let pid = selectedPID,
                    let proc = all.first(where: { $0.pid == pid })
                {
                    cpuHistory.append(proc.cpuPercent)
                    memHistory.append(proc.memoryMB)
                }

                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    func stopCollecting() {
        collectTask?.cancel()
        collectTask = nil
        isCollecting = false
    }

    func selectProcess(_ proc: ProcessInfo2) {
        selectedPID = proc.pid
        selectedName = proc.name
        cpuHistory = RollingBuffer(capacity: 900)
        memHistory = RollingBuffer(capacity: 900)
        cpuForecast = []
        memForecast = []
    }

    func predict(runner: TotoRunner) {
        guard runner.isModelLoaded, cpuHistory.count >= 64 else { return }
        isPredicting = true

        let vars = [cpuHistory.toArray(), memHistory.toArray()]

        if let forecasts = runner.forecastMultivariate(variables: vars, predictionLength: 128) {
            cpuForecast = forecasts[0].map { max(0, $0) }
            if forecasts.count > 1 { memForecast = forecasts[1].map { max(0, $0) } }
        }

        isPredicting = false
    }
}
