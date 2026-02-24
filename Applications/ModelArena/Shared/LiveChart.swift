import Charts
import SwiftUI

/// A data point for live charts.
struct MetricPoint: Identifiable {
    let id: Int
    let index: Int
    let value: Float
    let series: String
}

/// Reusable live chart with hover tooltip.
struct LiveChart: View {
    let title: String
    let unit: String
    let historical: [Float]
    let forecast: [Float]
    let height: CGFloat

    @State private var hoverIndex: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Spacer()
                if let latest = historical.last {
                    Text(formatValue(latest))
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .monospacedDigit()
                }
            }

            let points = buildPoints()

            Chart {
                if !forecast.isEmpty {
                    RuleMark(x: .value("Now", historical.count))
                        .foregroundStyle(.gray.opacity(0.3))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                }

                ForEach(points) { point in
                    LineMark(
                        x: .value("Sample", point.index),
                        y: .value(title, point.value),
                        series: .value("Series", point.series)
                    )
                    .foregroundStyle(point.series == "Forecast" ? Color.orange : Color.blue)
                    .lineStyle(StrokeStyle(
                        lineWidth: point.series == "Forecast" ? 2.5 : 1.5,
                        dash: point.series == "Forecast" ? [5, 3] : []))
                }

                if let idx = hoverIndex {
                    RuleMark(x: .value("Hover", idx))
                        .foregroundStyle(.white.opacity(0.2))
                }
            }
            .chartXAxis(.hidden)
            .chartYAxisLabel(unit)
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
                        .overlay(alignment: .topLeading) {
                            if let idx = hoverIndex,
                                let xPos = proxy.position(forX: idx)
                            {
                                tooltipView(at: idx)
                                    .offset(x: min(xPos - 40, geo.size.width - 120), y: 4)
                            }
                        }
                }
            }
        }
    }

    private func buildPoints() -> [MetricPoint] {
        var pts = [MetricPoint]()
        for (i, v) in historical.enumerated() {
            pts.append(MetricPoint(id: pts.count, index: i, value: v, series: "Historical"))
        }
        for (i, v) in forecast.enumerated() {
            pts.append(MetricPoint(
                id: pts.count, index: historical.count + i, value: v, series: "Forecast"))
        }
        return pts
    }

    private func tooltipView(at index: Int) -> some View {
        let value: Float
        let label: String
        if index < historical.count {
            value = historical[index]
            label = "Current"
        } else if index - historical.count < forecast.count {
            value = forecast[index - historical.count]
            label = "Forecast"
        } else {
            return AnyView(EmptyView())
        }
        return AnyView(
            VStack(alignment: .leading, spacing: 2) {
                Text(label).font(.caption2).foregroundStyle(.secondary)
                Text(formatValue(value))
                    .font(.caption).fontWeight(.semibold).monospacedDigit()
            }
            .padding(6)
            .background(RoundedRectangle(cornerRadius: 4).fill(.ultraThickMaterial))
        )
    }

    private func formatValue(_ v: Float) -> String {
        if unit == "%" { return String(format: "%.1f%%", v) }
        if unit == "W" { return String(format: "%.1fW", v) }
        if unit == "MB" { return String(format: "%.0f MB", v) }
        if unit == "MB/s" { return String(format: "%.1f MB/s", v) }
        return String(format: "%.1f %@", v, unit)
    }
}
