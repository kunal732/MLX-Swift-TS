import SwiftUI

@main
struct TotoMonitorApp: App {
    @State private var runner = TotoRunner()
    @State private var systemVM = SystemMetricsVM()
    @State private var batteryVM = BatteryVM()
    @State private var profilerVM = AppProfilerVM()

    var body: some Scene {
        WindowGroup {
            VStack(spacing: 0) {
                // Model status bar
                HStack(spacing: 12) {
                    Image(systemName: "brain")
                        .foregroundStyle(runner.isModelLoaded ? .green : .secondary)
                    Text(runner.statusMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    if runner.isLoading && runner.loadProgress > 0 {
                        ProgressView(value: runner.loadProgress)
                            .frame(width: 100)
                    }

                    Spacer()

                    if !runner.isModelLoaded {
                        Button("Load Toto Model") {
                            Task { await runner.loadModel() }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                        .disabled(runner.isLoading)
                    } else {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                        Text("Toto Ready")
                            .font(.caption)
                            .foregroundStyle(.green)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(.bar)

                Divider()

                TabView {
                    SystemMetricsView(vm: systemVM, runner: runner)
                        .tabItem {
                            Label("System", systemImage: "gauge.with.dots.needle.33percent")
                        }

                    BatteryView(vm: batteryVM, runner: runner)
                        .tabItem {
                            Label("Battery", systemImage: "battery.75percent")
                        }

                    AppProfilerView(vm: profilerVM, runner: runner)
                        .tabItem {
                            Label("App Profiler", systemImage: "app.badge")
                        }
                }
            }
        }
        .defaultSize(width: 1000, height: 700)
    }
}
