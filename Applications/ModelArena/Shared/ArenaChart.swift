import Charts
import SwiftUI

/// Colors assigned to each model in the arena.
private let modelColors: [Color] = [
    .orange, .green, .purple, .red, .cyan, .pink, .yellow, .mint,
]

/// A data point for the arena overlay chart.
private struct ArenaPoint: Identifiable {
    let id: Int
    let index: Int
    let value: Float
    let series: String
}

/// Multi-forecast overlay chart with sliding window.
///
/// - `historical`: recent actual samples up to "now" (sliding left every 2s)
/// - `groundTruth`: unused, kept for API compatibility
/// - `modelForecasts`: remaining forecast points (shrinks as predictions get consumed)
struct ArenaChart: View {
    let historical: [Float]
    let groundTruth: [Float]
    let modelForecasts: [(name: String, values: [Float])]
    let height: CGFloat

    @State private var hoverIndex: Int?

    private var nowIndex: Int { historical.count }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("CPU %")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Spacer()
                if let latest = historical.last {
                    Text(String(format: "%.1f%%", latest))
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .monospacedDigit()
                }
            }

            let points = buildPoints()
            let allValues = points.map(\.value)
            let yMin = max(0, (allValues.min() ?? 0) - 2)
            let yMax = min(100, (allValues.max() ?? 100) + 2)

            Chart {
                // "now" line — current moment, forecasts start here
                if !modelForecasts.isEmpty {
                    RuleMark(x: .value("Now", nowIndex))
                        .foregroundStyle(.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 3]))
                        .annotation(position: .top, alignment: .leading) {
                            Text("now")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                }

                ForEach(points) { point in
                    LineMark(
                        x: .value("Sample", point.index),
                        y: .value("CPU %", point.value),
                        series: .value("Series", point.series)
                    )
                    .foregroundStyle(colorForSeries(point.series))
                    .lineStyle(styleForSeries(point.series))
                }

                if let idx = hoverIndex {
                    RuleMark(x: .value("Hover", idx))
                        .foregroundStyle(.white.opacity(0.2))
                }
            }
            .chartXAxis(.hidden)
            .chartYAxisLabel("%")
            .chartYScale(domain: yMin...yMax)
            .frame(height: height)
            .chartOverlay { proxy in
                GeometryReader { geo in
                    Rectangle()
                        .fill(.clear)
                        .contentShape(Rectangle())
                        #if os(macOS)
                        .onContinuousHover { phase in
                            switch phase {
                            case .active(let loc):
                                hoverIndex = proxy.value(atX: loc.x) as Int?
                            case .ended:
                                hoverIndex = nil
                            }
                        }
                        #else
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    hoverIndex = proxy.value(atX: value.location.x) as Int?
                                }
                                .onEnded { _ in
                                    hoverIndex = nil
                                }
                        )
                        #endif
                }
            }

            // Legend
            if !modelForecasts.isEmpty {
                HStack(spacing: 12) {
                    legendItem(color: .blue, label: "Actual")
                    ForEach(Array(modelForecasts.enumerated()), id: \.offset) { idx, model in
                        legendItem(
                            color: modelColors[idx % modelColors.count],
                            label: model.name)
                    }
                }
                .font(.caption2)
            }
        }
    }

    private func legendItem(color: Color, label: String) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            Text(label)
                .foregroundStyle(.secondary)
        }
    }

    private func colorForSeries(_ series: String) -> Color {
        if series == "Historical" { return .blue }
        if let idx = modelForecasts.firstIndex(where: { $0.name == series }) {
            return modelColors[idx % modelColors.count]
        }
        return .gray
    }

    private func styleForSeries(_ series: String) -> StrokeStyle {
        if series == "Historical" {
            return StrokeStyle(lineWidth: 1.5)
        }
        return StrokeStyle(lineWidth: 2.0, dash: [5, 3])
    }

    private func buildPoints() -> [ArenaPoint] {
        var pts = [ArenaPoint]()

        // Actual data up to now
        for (i, v) in historical.enumerated() {
            pts.append(ArenaPoint(id: pts.count, index: i, value: v, series: "Historical"))
        }

        // Each model's remaining forecast (starts at "now", shrinks over time)
        for model in modelForecasts {
            for (i, v) in model.values.enumerated() {
                pts.append(ArenaPoint(
                    id: pts.count,
                    index: nowIndex + i,
                    value: v,
                    series: model.name))
            }
        }

        return pts
    }
}
