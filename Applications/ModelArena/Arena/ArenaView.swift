import SwiftUI

/// Colors assigned to each model for consistency with ArenaChart.
private let modelCardColors: [Color] = [
    .orange, .green, .purple, .red, .cyan, .pink, .yellow, .mint,
]

#if os(iOS)
/// Human-readable thermal state labels.
private func thermalLabel(_ value: Float) -> String {
    switch Int(value) {
    case 0: return "Nominal"
    case 1: return "Fair"
    case 2: return "Serious"
    case 3: return "Critical"
    default: return "?"
    }
}

private func thermalColor(_ value: Float) -> Color {
    switch Int(value) {
    case 0: return .green
    case 1: return .yellow
    case 2: return .orange
    case 3: return .red
    default: return .secondary
    }
}
#endif

struct ArenaView: View {
    @Bindable var vm: ArenaVM
    @Bindable var runner: ArenaRunner

    #if os(iOS)
    private let columns = [
        GridItem(.flexible(), spacing: 12),
    ]
    #else
    private let columns = [
        GridItem(.flexible(), spacing: 12),
        GridItem(.flexible(), spacing: 12),
    ]
    #endif

    var body: some View {
        ZStack {
            TimelineView(.periodic(from: .now, by: 2)) { _ in
                ScrollView {
                    VStack(spacing: 16) {
                        topBar
                        arenaChart
                        modelCards
                        if !runner.slots.isEmpty {
                            leaderboard
                        }
                    }
                    .padding()
                }
            }
            .onAppear {
                vm.autoRunner = runner
                vm.startCollecting()
            }
            .onDisappear { vm.stopCollecting() }

            #if os(iOS)
            if runner.isLoading {
                downloadOverlay
            }
            #endif
        }
    }

    #if os(iOS)
    // MARK: - Download Overlay

    private var downloadOverlay: some View {
        ZStack {
            Color.black.opacity(0.45)
                .ignoresSafeArea()

            VStack(spacing: 20) {
                Image(systemName: "arrow.down.circle.fill")
                    .font(.system(size: 44))
                    .foregroundStyle(.white)

                VStack(spacing: 6) {
                    Text("Downloading Models")
                        .font(.headline)
                        .foregroundStyle(.white)

                    Text(runner.loadingStatus)
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.8))
                        .multilineTextAlignment(.center)
                        .lineLimit(2)
                }

                VStack(spacing: 8) {
                    if runner.loadProgress > 0 {
                        ProgressView(value: runner.loadProgress)
                            .tint(.white)
                            .frame(width: 240)

                        Text(String(format: "%.0f%%", runner.loadProgress * 100))
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.7))
                            .monospacedDigit()
                    } else {
                        ProgressView()
                            .tint(.white)
                    }

                    if runner.totalModelsToLoad > 1 {
                        Text("Model \(runner.currentModelIndex) of \(runner.totalModelsToLoad)")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.7))
                    }
                }
            }
            .padding(32)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(.ultraThinMaterial)
            )
            .padding(.horizontal, 40)
        }
    }
    #endif

    // MARK: - Top Bar

    private var topBar: some View {
        HStack(spacing: 16) {
            // Live CPU
            HStack(spacing: 6) {
                Image(systemName: "cpu")
                    .foregroundStyle(.blue)
                Text(String(format: "CPU %.1f%%", vm.latestSnapshot.cpuPercent))
                    .font(.headline)
                    .monospacedDigit()
            }

            Divider().frame(height: 20)

            #if os(iOS)
            // Thermal state
            HStack(spacing: 4) {
                Image(systemName: "thermometer.medium")
                    .foregroundStyle(thermalColor(vm.latestSnapshot.thermalState))
                Text(thermalLabel(vm.latestSnapshot.thermalState))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Divider().frame(height: 20)

            // Battery
            HStack(spacing: 4) {
                Image(systemName: "battery.100percent")
                    .foregroundStyle(.green)
                Text(String(format: "%.0f%%", vm.latestSnapshot.batteryLevel))
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            #else
            // Sample count
            HStack(spacing: 4) {
                Image(systemName: "waveform.path")
                    .foregroundStyle(.secondary)
                Text("\(vm.sampleCount) samples")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Divider().frame(height: 20)

            // Models loaded
            HStack(spacing: 4) {
                Image(systemName: "brain")
                    .foregroundStyle(runner.loadedCount > 0 ? .green : .secondary)
                Text("\(runner.loadedCount) model(s)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            #endif

            Spacer()

            // Run All button
            Button {
                vm.predict(runner: runner)
            } label: {
                HStack(spacing: 4) {
                    if vm.isPredicting {
                        ProgressView()
                            .controlSize(.small)
                    }
                    Text("Run All")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(!vm.hasEnoughSamples || runner.slots.isEmpty || vm.isPredicting)
            #if os(macOS)
            .help(
                !vm.hasEnoughSamples
                    ? "Need at least 64 samples (~2 min)"
                    : runner.slots.isEmpty ? "Load models first" : "Run all models"
            )
            #endif
        }
        .padding(.horizontal, 4)
    }

    // MARK: - Main Chart

    /// Number of historical samples visible in the sliding window.
    private let visibleHistory = 150

    private var arenaChart: some View {
        // Access tick to ensure re-render on every 2s sample
        let currentTick = vm.tick
        let _ = currentTick

        let fullHistory = vm.cpuHistory.toArray()

        // Sliding window of recent actual data — always up to "right now"
        let windowHistory = Array(fullHistory.suffix(visibleHistory))

        // How many ticks have elapsed since the forecast was made
        let elapsed = vm.forecastTick > 0
            ? max(0, currentTick - vm.forecastTick)
            : 0

        // Trim consumed points from the front of each forecast
        // so the remaining forecast starts at "now" and shifts left each tick
        let forecasts: [(name: String, values: [Float])] = runner.slots
            .filter { !$0.forecast.isEmpty }
            .compactMap { slot in
                let remaining = Array(slot.forecast.dropFirst(elapsed))
                guard !remaining.isEmpty else { return nil }
                return (name: slot.name, values: remaining)
            }

        // Auto-refresh when forecast is almost consumed
        let _ = Task { @MainActor in
            if vm.forecastTick > 0,
               !vm.isPredicting,
               elapsed >= 120,
               let r = vm.autoRunner, !r.slots.isEmpty
            {
                vm.predict(runner: r)
            }
        }

        return ArenaChart(
            historical: windowHistory,
            groundTruth: [],
            modelForecasts: forecasts,
            height: 250
        )
        .padding()
        .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.3)))
    }

    // MARK: - Model Cards

    private var modelCards: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Models")
                .font(.headline)

            if runner.slots.isEmpty {
                HStack {
                    Spacer()
                    VStack(spacing: 8) {
                        Image(systemName: "brain")
                            .font(.largeTitle)
                            .foregroundStyle(.secondary)
                        Text("No models loaded")
                            .foregroundStyle(.secondary)
                        #if os(iOS)
                        #if targetEnvironment(simulator)
                        Text("Models require a physical device (MLX needs Metal GPU)")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                        #else
                        Text("Tap \"HuggingFace\" to download or \"Files\" to load local models")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                        #endif
                        #else
                        Text("Click \"Load Models\" to select a directory of converted models")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                        #endif
                    }
                    .padding(.vertical, 40)
                    Spacer()
                }
            } else {
                LazyVGrid(columns: columns, spacing: 12) {
                    ForEach(Array(runner.slots.enumerated()), id: \.element.id) { idx, slot in
                        modelCard(slot: slot, colorIndex: idx)
                    }
                }
            }
        }
    }

    private func modelCard(slot: ModelSlot, colorIndex: Int) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                // Status dot
                Circle()
                    .fill(modelCardColors[colorIndex % modelCardColors.count])
                    .frame(width: 8, height: 8)

                Text(slot.name)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .lineLimit(1)

                Spacer()

                // Architecture badge
                Text(slot.modelType)
                    .font(.caption2)
                    .fontWeight(.medium)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(
                        Capsule()
                            .fill(modelCardColors[colorIndex % modelCardColors.count].opacity(0.2))
                    )
            }

            HStack(spacing: 16) {
                // Inference time
                VStack(alignment: .leading, spacing: 2) {
                    Text("Inference")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if slot.inferenceMs > 0 {
                        Text(String(format: "%.0f ms", slot.inferenceMs))
                            .font(.caption)
                            .fontWeight(.semibold)
                            .monospacedDigit()
                    } else {
                        Text("--")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                }

                // Live MASE
                VStack(alignment: .leading, spacing: 2) {
                    Text("MASE")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if let mase = slot.liveMASE {
                        Text(String(format: "%.3f", mase))
                            .font(.caption)
                            .fontWeight(.semibold)
                            .monospacedDigit()
                            .foregroundStyle(mase < 1.0 ? .green : mase < 2.0 ? .orange : .red)
                    } else {
                        Text("--")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                }

                Spacer()

                // Forecast status
                if !slot.forecast.isEmpty {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.caption)
                } else {
                    Image(systemName: "circle.dashed")
                        .foregroundStyle(.tertiary)
                        .font(.caption)
                }
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.3)))
    }

    // MARK: - Leaderboard

    private var leaderboard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Leaderboard")
                .font(.headline)

            let sorted = runner.slots.enumerated()
                .sorted { a, b in
                    let aMASE = a.element.liveMASE ?? .infinity
                    let bMASE = b.element.liveMASE ?? .infinity
                    return aMASE < bMASE
                }

            let maxMASE = sorted.compactMap({ $0.element.liveMASE }).max() ?? 1.0
            let barMax = max(maxMASE, 1.0)

            ForEach(Array(sorted.enumerated()), id: \.element.offset) { rank, entry in
                let slot = entry.element
                let colorIdx = entry.offset

                HStack(spacing: 8) {
                    Text("#\(rank + 1)")
                        .font(.caption)
                        .fontWeight(.bold)
                        .frame(width: 28)
                        .foregroundStyle(rank == 0 ? .yellow : .secondary)

                    Circle()
                        .fill(modelCardColors[colorIdx % modelCardColors.count])
                        .frame(width: 6, height: 6)

                    Text(slot.name)
                        .font(.caption)
                        .lineLimit(1)
                        .frame(width: 100, alignment: .leading)

                    if let mase = slot.liveMASE {
                        GeometryReader { geo in
                            RoundedRectangle(cornerRadius: 3)
                                .fill(modelCardColors[colorIdx % modelCardColors.count]
                                    .opacity(0.6))
                                .frame(width: max(4, geo.size.width * CGFloat(mase / barMax)))
                        }
                        .frame(height: 12)

                        Text(String(format: "%.3f", mase))
                            .font(.caption2)
                            .monospacedDigit()
                            .frame(width: 50, alignment: .trailing)
                    } else {
                        Spacer()
                        Text("pending")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
            }
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.3)))
    }
}
