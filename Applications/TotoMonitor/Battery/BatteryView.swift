import SwiftUI

struct BatteryView: View {
    @Bindable var vm: BatteryVM
    var runner: TotoRunner

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let snap = vm.latestSnapshot {
                    HStack(spacing: 20) {
                        BatteryGauge(percent: snap.percent, isCharging: snap.isCharging)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(String(format: "%.0f%%", snap.percent))
                                .font(.largeTitle)
                                .fontWeight(.bold)
                            Text(
                                snap.isCharging
                                    ? "Charging" : snap.isPluggedIn ? "Plugged In" : "On Battery"
                            )
                            .foregroundStyle(.secondary)
                        }
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Image(systemName: "bolt.fill").foregroundStyle(.yellow)
                                Text(String(format: "%.1f W", snap.watts))
                                    .fontWeight(.semibold)
                            }
                            if let mins = snap.minutesRemaining {
                                Text("System estimate: \(mins / 60)h \(mins % 60)m")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            if let totoEst = vm.predictedTimeToEmpty {
                                Text("Toto predicts: \(totoEst)")
                                    .font(.caption).foregroundStyle(.orange)
                            }
                        }
                        Spacer()
                        Button(action: { vm.predict(runner: runner) }) {
                            Label(
                                vm.isPredicting ? "Predicting..." : "Predict",
                                systemImage: "chart.line.uptrend.xyaxis")
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.orange)
                        .disabled(
                            !runner.isModelLoaded || vm.isPredicting
                                || vm.percentHistory.count < 64)
                    }
                } else {
                    Text("No battery detected")
                        .foregroundStyle(.secondary)
                }

                LiveChart(
                    title: "Battery Level",
                    unit: "%",
                    historical: vm.percentHistory.toArray(),
                    forecast: vm.percentForecast,
                    height: 200
                )

                LiveChart(
                    title: "Power Draw",
                    unit: "W",
                    historical: vm.wattsHistory.toArray(),
                    forecast: vm.wattsForecast,
                    height: 160
                )

                Text(
                    "Collecting every 10s. Need 64+ samples (~11 min) before prediction. Forecast shows ~20 min ahead."
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

struct BatteryGauge: View {
    let percent: Float
    let isCharging: Bool

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 4)
                .stroke(Color.primary.opacity(0.3), lineWidth: 2)
                .frame(width: 50, height: 24)

            RoundedRectangle(cornerRadius: 2)
                .fill(percent > 20 ? Color.green : Color.red)
                .frame(width: max(2, 46 * CGFloat(percent / 100)), height: 20)
                .frame(width: 46, alignment: .leading)

            if isCharging {
                Image(systemName: "bolt.fill")
                    .font(.caption2)
                    .foregroundStyle(.white)
            }
        }
    }
}
