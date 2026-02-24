import SwiftUI

struct AppProfilerView: View {
    @Bindable var vm: AppProfilerVM
    var runner: TotoRunner

    var body: some View {
        HSplitView {
            // Process list
            VStack(alignment: .leading, spacing: 0) {
                Text("Running Processes")
                    .font(.headline)
                    .padding(.horizontal)
                    .padding(.top, 8)

                List(vm.processes, id: \.pid, selection: Binding(
                    get: { vm.selectedPID },
                    set: { pid in
                        if let pid, let proc = vm.processes.first(where: { $0.pid == pid }) {
                            vm.selectProcess(proc)
                        }
                    }
                )) { proc in
                    HStack {
                        Text(proc.name)
                            .lineLimit(1)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Text(String(format: "%.1f%%", proc.cpuPercent))
                            .font(.caption)
                            .monospacedDigit()
                            .frame(width: 50, alignment: .trailing)
                        Text(String(format: "%.0f MB", proc.memoryMB))
                            .font(.caption)
                            .monospacedDigit()
                            .frame(width: 70, alignment: .trailing)
                    }
                    .tag(proc.pid)
                }
                .listStyle(.bordered)
            }
            .frame(minWidth: 280, maxWidth: 320)

            // Selected process detail
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if vm.selectedPID != nil {
                        HStack {
                            Text(vm.selectedName)
                                .font(.title2)
                                .fontWeight(.bold)
                            Text("PID \(vm.selectedPID ?? 0)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
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
                                    || vm.cpuHistory.count < 64)
                        }

                        if !vm.memForecast.isEmpty, let peak = vm.memForecast.max(),
                            let current = vm.memHistory.latest
                        {
                            if peak > current * 1.5 {
                                HStack {
                                    Image(systemName: "exclamationmark.triangle.fill")
                                        .foregroundStyle(.yellow)
                                    Text(
                                        String(
                                            format:
                                                "%@ trending from %.0f MB to %.0f MB",
                                            vm.selectedName, current, peak))
                                        .font(.callout)
                                }
                                .padding(10)
                                .background(
                                    RoundedRectangle(cornerRadius: 8).fill(
                                        .yellow.opacity(0.15)))
                            }
                        }

                        LiveChart(
                            title: "CPU Usage",
                            unit: "%",
                            historical: vm.cpuHistory.last(300),
                            forecast: vm.cpuForecast,
                            height: 180
                        )

                        LiveChart(
                            title: "Memory",
                            unit: "MB",
                            historical: vm.memHistory.last(300),
                            forecast: vm.memForecast,
                            height: 180
                        )

                        Text(
                            "Select a process and wait ~2 min for data. Click Predict for Toto forecast."
                        )
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                    } else {
                        VStack(spacing: 10) {
                            Image(systemName: "app.dashed")
                                .font(.system(size: 40))
                                .foregroundStyle(.secondary)
                            Text("Select a process to track")
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding(.top, 100)
                    }
                }
                .padding()
            }
        }
        .onAppear { vm.startCollecting() }
        .onDisappear { vm.stopCollecting() }
    }
}
