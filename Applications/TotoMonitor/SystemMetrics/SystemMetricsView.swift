import SwiftUI

struct SystemMetricsView: View {
    @Bindable var vm: SystemMetricsVM
    var runner: TotoRunner

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Live stats bar
                HStack(spacing: 20) {
                    MetricBadge(
                        label: "CPU", value: String(format: "%.1f%%", vm.latestSnapshot.cpuPercent),
                        color: .blue)
                    MetricBadge(
                        label: "Memory",
                        value: String(
                            format: "%.0f / %.0f MB", vm.latestSnapshot.memoryUsedMB,
                            vm.latestSnapshot.memoryTotalMB),
                        color: .green)
                    MetricBadge(
                        label: "GPU", value: String(format: "%.0f%%", vm.latestSnapshot.gpuPercent),
                        color: .purple)
                    MetricBadge(
                        label: "Disk Read",
                        value: String(format: "%.1f MB/s", vm.latestSnapshot.diskReadMBps),
                        color: .orange)
                    MetricBadge(
                        label: "Net In",
                        value: String(format: "%.1f MB/s", vm.latestSnapshot.netInMBps),
                        color: .cyan)
                    Spacer()
                    Button(action: { vm.predict(runner: runner) }) {
                        Label(
                            vm.isPredicting ? "Predicting..." : "Predict",
                            systemImage: "chart.line.uptrend.xyaxis")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    .disabled(!runner.isModelLoaded || vm.isPredicting || vm.cpuHistory.count < 64)
                }

                if let alert = vm.alert {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.yellow)
                        Text(alert)
                            .font(.callout)
                            .fontWeight(.medium)
                    }
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: 8).fill(.yellow.opacity(0.15)))
                }

                // Charts
                LiveChart(
                    title: "CPU Usage",
                    unit: "%",
                    historical: vm.cpuHistory.last(300),
                    forecast: vm.cpuForecast,
                    height: 180
                )

                LiveChart(
                    title: "Memory Usage",
                    unit: "%",
                    historical: vm.memHistory.last(300),
                    forecast: vm.memForecast,
                    height: 180
                )

                LiveChart(
                    title: "GPU Usage",
                    unit: "%",
                    historical: vm.gpuHistory.last(300),
                    forecast: vm.gpuForecast,
                    height: 140
                )

                HStack(spacing: 16) {
                    LiveChart(
                        title: "Disk Read",
                        unit: "MB/s",
                        historical: vm.diskReadHistory.last(300),
                        forecast: [],
                        height: 120
                    )
                    LiveChart(
                        title: "Network In",
                        unit: "MB/s",
                        historical: vm.netInHistory.last(300),
                        forecast: [],
                        height: 120
                    )
                }

                Text(
                    "Collecting every 2s. Need 64+ samples (~2 min) before prediction. Forecast shows ~4 min ahead."
                )
                .font(.caption)
                .foregroundStyle(.tertiary)
            }
            .padding()
        }
        .onAppear { vm.startCollecting() }
        .onDisappear { vm.stopCollecting() }
    }
}

struct MetricBadge: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.caption).fontWeight(.bold).monospacedDigit()
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(RoundedRectangle(cornerRadius: 6).fill(color.opacity(0.1)))
    }
}
