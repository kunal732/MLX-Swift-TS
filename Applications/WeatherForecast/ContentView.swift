import Charts
import SwiftUI

struct ContentView: View {
    @State private var viewModel = ForecastViewModel()

    private let dayTimeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "E ha"
        return f
    }()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                headerSection
                Divider()
                inputSection
                if !viewModel.temperaturePoints.isEmpty {
                    forecastSummarySection
                    temperatureChartSection
                    if !viewModel.precipPoints.isEmpty {
                        precipChartSection
                    }
                    if !viewModel.snowfallPoints.isEmpty {
                        snowfallChartSection
                    }
                    if !viewModel.weatherServiceEvents.isEmpty {
                        weatherServiceEventsSection
                    }
                    if !viewModel.precipEvents.isEmpty {
                        totoEventsSection
                    }
                    accuracySection
                } else {
                    placeholderSection
                }
                Divider()
                howItWorksSection
            }
            .padding(24)
        }
        .frame(minWidth: 750, minHeight: 650)
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Toto Weather Forecast")
                .font(.largeTitle)
                .fontWeight(.bold)

            Text(
                "Datadog's 151M-parameter time series foundation model, running locally on Apple Silicon via MLX"
            )
            .font(.callout)
            .foregroundStyle(.secondary)
        }
    }

    // MARK: - Input

    private var inputSection: some View {
        VStack(spacing: 8) {
            HStack(spacing: 12) {
                TextField("Zip code", text: $viewModel.zipCode)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
                    .onSubmit {
                        if viewModel.modelLoaded { Task { await viewModel.runForecast() } }
                    }

                if !viewModel.modelLoaded {
                    Button(action: { Task { await viewModel.loadModel() } }) {
                        Label("Load Model", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isLoading)
                } else {
                    Button(action: { Task { await viewModel.runForecast() } }) {
                        Label("3-Day Forecast", systemImage: "cloud.sun")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                    .disabled(viewModel.isLoading || viewModel.zipCode.isEmpty)
                }

                HStack(spacing: 6) {
                    if viewModel.isLoading {
                        ProgressView()
                            .scaleEffect(0.6)
                            .frame(width: 14, height: 14)
                    }
                    Text(viewModel.statusMessage)
                        .font(.caption)
                        .foregroundStyle(
                            viewModel.statusMessage.contains("Failed") ? .red : .secondary
                        )
                        .lineLimit(1)
                }

                Spacer()

                if let elapsed = viewModel.elapsedTime {
                    Text(elapsed)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .monospacedDigit()
                }
            }

            if viewModel.isLoading && viewModel.downloadProgress > 0 {
                ProgressView(value: viewModel.downloadProgress)
                    .progressViewStyle(.linear)
            }
        }
    }

    // MARK: - Forecast Summary

    private var forecastSummarySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("3-Day Forecast for \(viewModel.locationName)")
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
            }

            HStack(spacing: 20) {
                if let high = viewModel.forecastHighTemp {
                    StatBadge(
                        icon: "thermometer.sun", label: "High",
                        value: String(format: "%.0f\u{00B0}F", high), color: .orange)
                }
                if let low = viewModel.forecastLowTemp {
                    StatBadge(
                        icon: "thermometer.snowflake", label: "Low",
                        value: String(format: "%.0f\u{00B0}F", low), color: .blue)
                }
                if let rain = viewModel.totalRainInches {
                    StatBadge(
                        icon: "cloud.rain", label: "Rain",
                        value: String(format: "%.2f\"", rain), color: .cyan)
                }
                if let snow = viewModel.totalSnowInches {
                    StatBadge(
                        icon: "cloud.snow", label: "Snow",
                        value: String(format: "%.1f\"", snow), color: .indigo)
                }
                if let wsSnow = viewModel.weatherServiceSnow {
                    StatBadge(
                        icon: "cloud.snow", label: "WS Snow",
                        value: String(format: "%.1f\"", wsSnow), color: .purple)
                }
                if viewModel.totalRainInches == nil && viewModel.totalSnowInches == nil
                    && viewModel.weatherServiceSnow == nil
                {
                    StatBadge(
                        icon: "sun.max", label: "Precip", value: "None", color: .green)
                }
            }
        }
    }

    // MARK: - Temperature Chart

    private var temperatureChartSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Temperature")
                .font(.headline)

            InteractiveChart(
                points: viewModel.temperaturePoints,
                yLabel: "Temperature (\u{00B0}F)",
                unit: "\u{00B0}F",
                showBoundary: true,
                boundaryHour: viewModel.temperaturePoints.first(where: {
                    $0.kind == .totoForecast
                })?.hour ?? 0
            )
            .frame(height: 280)
        }
    }

    // MARK: - Precipitation Chart

    private var precipChartSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Precipitation")
                .font(.headline)

            InteractiveChart(
                points: viewModel.precipPoints,
                yLabel: "Precipitation (in)",
                unit: "\"",
                showBoundary: true,
                boundaryHour: viewModel.precipPoints.first(where: {
                    $0.kind == .totoForecast
                })?.hour ?? 0
            )
            .frame(height: 180)
        }
    }

    // MARK: - Snowfall Chart

    private var snowfallChartSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Snowfall Accumulation (Weather Service)")
                    .font(.headline)
                Text("Cumulative total from Open-Meteo — Toto cannot predict snow events")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            InteractiveChart(
                points: viewModel.snowfallPoints,
                yLabel: "Total Snow (in)",
                unit: "\"",
                showBoundary: true,
                boundaryHour: viewModel.snowfallPoints.first(where: {
                    $0.kind == .actual
                })?.hour ?? 0
            )
            .frame(height: 160)
        }
    }

    // MARK: - Weather Service Events

    private var weatherServiceEventsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Weather Service Forecast (Open-Meteo)")
                .font(.headline)

            ForEach(viewModel.weatherServiceEvents) { event in
                precipEventRow(event)
            }
        }
    }

    // MARK: - Toto Events

    private var totoEventsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Toto Prediction (Statistical)")
                .font(.headline)

            ForEach(viewModel.precipEvents) { event in
                precipEventRow(event)
            }
        }
    }

    // MARK: - Shared Event Row

    private func precipEventRow(_ event: PrecipEvent) -> some View {
        HStack(spacing: 12) {
            Image(
                systemName: event.type == "Snow"
                    ? "cloud.snow.fill"
                    : event.type == "Rain" ? "cloud.rain.fill" : "cloud.sleet.fill"
            )
            .foregroundStyle(
                event.type == "Snow" ? .indigo : event.type == "Rain" ? .cyan : .purple
            )
            .font(.title3)
            .frame(width: 30)

            VStack(alignment: .leading, spacing: 2) {
                Text(event.type)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                if let start = event.startDate, let end = event.endDate {
                    Text(
                        "\(dayTimeFormatter.string(from: start)) \u{2013} \(dayTimeFormatter.string(from: end))"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }

            Spacer()

            Text(String(format: "%.1f\"", event.totalInches))
                .font(.subheadline)
                .fontWeight(.medium)
                .monospacedDigit()
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary))
    }

    // MARK: - Accuracy Note

    private var accuracySection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Toto vs Traditional Weather Models")
                .font(.headline)

            Text("""
                The green dashed line shows the weather service forecast from Open-Meteo (which uses physics-based \
                models like ECMWF and GFS). Toto is a general-purpose time series foundation model doing \
                zero-shot statistical forecasting \u{2014} it has no knowledge of weather physics, satellite \
                data, or atmospheric models. It only sees the raw numbers over time.
                """)
                .font(.caption)
                .foregroundStyle(.secondary)

            Text("""
                Toto tends to predict smoother, mean-reverting values. It captures the overall trend and \
                daily cycle well, but misses sudden changes that physics models can anticipate from pressure \
                systems and fronts. This is expected \u{2014} the impressive part is that it produces \
                reasonable forecasts from numbers alone, with no domain-specific training.
                """)
                .font(.caption)
                .foregroundStyle(.secondary)

            if let mae = viewModel.temperatureMAE {
                HStack(spacing: 16) {
                    StatBadge(
                        icon: "chart.bar", label: "Avg Error",
                        value: String(format: "%.1f\u{00B0}F", mae), color: .purple)
                    Text(
                        "Mean absolute difference between Toto and weather service forecasts"
                    )
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                }
                .padding(.top, 4)
            }
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 10).fill(.quaternary.opacity(0.5)))
    }

    // MARK: - Placeholder

    private var placeholderSection: some View {
        RoundedRectangle(cornerRadius: 12)
            .fill(Color.gray.opacity(0.06))
            .frame(height: 260)
            .overlay(
                VStack(spacing: 10) {
                    Image(systemName: "cloud.sun")
                        .font(.system(size: 40))
                        .foregroundStyle(.secondary)
                    Text("Enter a zip code, load the model, and run a forecast")
                        .foregroundStyle(.secondary)
                    Text(
                        "Toto analyzes 7 days of real hourly weather and predicts the next 3 days"
                    )
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                }
            )
    }

    // MARK: - How It Works

    private var howItWorksSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("How It Works")
                .font(.headline)

            HStack(alignment: .top, spacing: 24) {
                StepCard(
                    icon: "location.circle", number: 1, title: "Fetch Data",
                    detail:
                        "Your zip code is geocoded, then 7 days of hourly weather is fetched from Open-Meteo (temp, precip, humidity, wind)."
                )
                StepCard(
                    icon: "square.grid.3x3", number: 2, title: "Patch & Normalize",
                    detail:
                        "The time series is split into 64-step patches and normalized causally so no future data leaks."
                )
                StepCard(
                    icon: "brain", number: 3, title: "Toto Transformer",
                    detail:
                        "12 transformer layers with RoPE process the patches on your Mac's GPU via MLX. Temperature and precipitation are forecast independently."
                )
                StepCard(
                    icon: "chart.bar.xaxis", number: 4, title: "Predict",
                    detail:
                        "The model outputs a 24-component Student-t mixture per timestep. Rain vs snow is classified by whether predicted temp is above/below freezing."
                )
            }
        }
    }
}

// MARK: - Interactive Chart with Hover Tooltip

struct InteractiveChart: View {
    let points: [ChartPoint]
    let yLabel: String
    let unit: String
    let showBoundary: Bool
    let boundaryHour: Int

    @State private var hoverHour: Int?

    private var hoveredPoints: [ChartPoint] {
        guard let hour = hoverHour else { return [] }
        return points.filter { $0.hour == hour }
    }

    var body: some View {
        Chart {
            if showBoundary && boundaryHour > 0 {
                RuleMark(x: .value("Now", boundaryHour))
                    .foregroundStyle(.gray.opacity(0.4))
                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [4, 4]))
                    .annotation(position: .top, alignment: .leading) {
                        Text(" Now")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
            }

            ForEach(points) { point in
                LineMark(
                    x: .value("Hour", point.hour),
                    y: .value("Value", point.temperature),
                    series: .value("Series", point.kind.rawValue)
                )
                .foregroundStyle(by: .value("Series", point.kind.rawValue))
                .lineStyle(StrokeStyle(
                    lineWidth: point.kind == .totoForecast ? 2.5 : 1.5,
                    dash: point.kind == .actual ? [4, 3] : []))
            }

            // Hover indicator
            if let hour = hoverHour {
                RuleMark(x: .value("Hover", hour))
                    .foregroundStyle(.white.opacity(0.3))
                    .lineStyle(StrokeStyle(lineWidth: 1))
            }
        }
        .chartForegroundStyleScale([
            "Historical": Color.blue,
            "Toto Forecast": Color.orange,
            "Weather Service": Color.green.opacity(0.7),
        ])
        .chartXAxisLabel("Hours")
        .chartYAxisLabel(yLabel)
        .chartLegend(position: .bottom)
        .chartOverlay { proxy in
            GeometryReader { geo in
                Rectangle()
                    .fill(.clear)
                    .contentShape(Rectangle())
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(let location):
                            if let hour: Int = proxy.value(atX: location.x) {
                                hoverHour = hour
                            }
                        case .ended:
                            hoverHour = nil
                        }
                    }
                    .overlay(alignment: .topLeading) {
                        if !hoveredPoints.isEmpty, let hour = hoverHour,
                            let xPos = proxy.position(forX: hour)
                        {
                            tooltipView
                                .offset(
                                    x: min(
                                        xPos - 50,
                                        geo.size.width - 160),
                                    y: 4)
                        }
                    }
            }
        }
    }

    private var tooltipView: some View {
        VStack(alignment: .leading, spacing: 3) {
            if let first = hoveredPoints.first, let date = first.date {
                let fmt = DateFormatter()
                Text(
                    {
                        let f = DateFormatter()
                        f.dateFormat = "EEE MMM d, h:mm a"
                        return f.string(from: date)
                    }()
                )
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
            ForEach(hoveredPoints) { point in
                HStack(spacing: 4) {
                    Circle()
                        .fill(colorForKind(point.kind))
                        .frame(width: 6, height: 6)
                    Text(point.kind.rawValue)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(String(format: "%.1f%@", point.temperature, unit))
                        .font(.caption)
                        .fontWeight(.semibold)
                        .monospacedDigit()
                }
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(.ultraThickMaterial)
                .shadow(radius: 4)
        )
        .frame(width: 150)
    }

    private func colorForKind(_ kind: ChartPoint.PointKind) -> Color {
        switch kind {
        case .historical: return .blue
        case .totoForecast: return .orange
        case .actual: return .green
        }
    }
}

// MARK: - Helper Views

struct StatBadge: View {
    let icon: String
    let label: String
    let value: String
    let color: Color

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .foregroundStyle(color)
            VStack(alignment: .leading, spacing: 1) {
                Text(label)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(value)
                    .font(.callout)
                    .fontWeight(.semibold)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(RoundedRectangle(cornerRadius: 8).fill(color.opacity(0.1)))
    }
}

struct StepCard: View {
    let icon: String
    let number: Int
    let title: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .foregroundStyle(.orange)
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.semibold)
            }
            Text(detail)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
