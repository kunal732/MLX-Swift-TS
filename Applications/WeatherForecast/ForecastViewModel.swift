import Foundation
import MLX
import MLXTimeSeries

// MARK: - Chart Data

struct ChartPoint: Identifiable {
    let id: Int
    let hour: Int
    let date: Date?
    let temperature: Float
    let kind: PointKind

    enum PointKind: String {
        case historical = "Historical"
        case totoForecast = "Toto Forecast"
        case actual = "Weather Service"
    }
}

struct PrecipEvent: Identifiable {
    let id = UUID()
    let type: String
    let startHour: Int
    let endHour: Int
    let totalInches: Float
    let startDate: Date?
    let endDate: Date?
}

// MARK: - View Model

@MainActor
@Observable
class ForecastViewModel {

    // Input
    var zipCode: String = ""

    // State
    var statusMessage: String = "Enter a zip code and load the model to get started"
    var isLoading: Bool = false
    var modelLoaded: Bool = false
    var downloadProgress: Double = 0
    var elapsedTime: String?

    // Location
    var locationName: String = ""

    // Chart data
    var temperaturePoints: [ChartPoint] = []
    var precipPoints: [ChartPoint] = []
    var snowfallPoints: [ChartPoint] = []
    var precipEvents: [PrecipEvent] = []
    var weatherServiceEvents: [PrecipEvent] = []

    // Forecast stats
    var forecastHighTemp: Float?
    var forecastLowTemp: Float?
    var totalRainInches: Float?
    var totalSnowInches: Float?
    var temperatureMAE: Float?

    // Internal
    var historicalCount: Int = 0
    var forecastCount: Int = 0

    private var forecaster: TotoForecaster?
    private var startTime: Date?
    private var weatherData: WeatherData?

    // MARK: - Load Model

    func loadModel() async {
        isLoading = true
        downloadProgress = 0
        startTime = Date()
        statusMessage = "Downloading Toto-Open-Base-1.0 from HuggingFace (~605 MB)..."

        do {
            forecaster = try await TotoForecaster.loadFromHub(
                id: "kunal732/Toto-Open-Base-1.0-MLX"
            ) { [weak self] progress in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.downloadProgress = progress.fractionCompleted
                    let mb = Int(progress.completedUnitCount / 1_000_000)
                    let totalMb = max(1, Int(progress.totalUnitCount / 1_000_000))
                    self.statusMessage = "Downloading model... \(mb)/\(totalMb) MB"
                    self.updateElapsed()
                }
            }
            modelLoaded = true
            updateElapsed()
            statusMessage = "Model loaded. Enter a zip code and click Forecast."
        } catch {
            statusMessage = "Failed to load model: \(error.localizedDescription)"
        }

        isLoading = false
        downloadProgress = 0
    }

    // MARK: - Run Forecast

    func runForecast() async {
        guard let forecaster else {
            statusMessage = "Load the model first"
            return
        }
        guard !zipCode.isEmpty else {
            statusMessage = "Enter a zip code"
            return
        }

        isLoading = true
        startTime = Date()

        // Step 1: Geocode and fetch weather
        statusMessage = "Looking up \(zipCode)..."
        do {
            let data = try await WeatherService.getWeatherData(zipCode: zipCode)
            self.weatherData = data
            self.locationName = data.locationName
            self.historicalCount = data.historicalCount
            statusMessage = "Fetched weather for \(data.locationName). Running Toto..."
            updateElapsed()
        } catch {
            statusMessage = "Failed to fetch weather: \(error.localizedDescription)"
            isLoading = false
            return
        }

        guard let data = weatherData else { return }

        let predictionLength = 72  // 3 days
        self.forecastCount = predictionLength
        let H = data.historicalCount

        // Step 2: Build multivariate input — 7 variables so the cross-variate
        // attention layer (layer 11) can learn relationships between them.
        // Variables: temp, precip, humidity, wind, pressure, cloud cover, dew point
        statusMessage = "Running Toto multivariate forecast (7 variables, 72 hrs)..."
        updateElapsed()

        let variables: [[Float]] = [
            Array(data.temperature.prefix(H)),
            Array(data.precipitation.prefix(H)),
            Array(data.humidity.prefix(H)),
            Array(data.windSpeed.prefix(H)),
            Array(data.pressure.prefix(H)),
            Array(data.cloudCover.prefix(H)),
            Array(data.dewPoint.prefix(H)),
        ]
        let V = variables.count
        let T = H

        // Build [1, V, T] tensor
        var flatValues = [Float]()
        for v in 0 ..< V {
            flatValues.append(contentsOf: variables[v])
        }
        let seriesArray = MLXArray(flatValues).reshaped(1, V, T)
        let maskArray = MLXArray.ones([1, V, T])
        // All variables share the same ID so they can attend to each other
        let idMask = MLXArray.zeros([1, V], type: Int32.self)

        let input = TimeSeriesInput(
            series: seriesArray,
            paddingMask: maskArray,
            idMask: idMask
        ).padded(toPatchSize: 64)

        let result = forecaster.forecast(input: input, predictionLength: predictionLength)

        // result.mean: [1, V, predictionLength] — extract each variable
        let forecastAll = result.mean.squeezed(axis: 0)  // [V, predictionLength]
        eval(forecastAll)

        let tempForecast = (0 ..< predictionLength).map {
            forecastAll[0, $0].item(Float.self)
        }
        let precipForecast = (0 ..< predictionLength).map {
            max(0, forecastAll[1, $0].item(Float.self))
        }

        // Step 4: Build charts
        buildTemperatureChart(data: data, tempForecast: tempForecast)
        buildPrecipChart(data: data, precipForecast: precipForecast)
        buildSnowfallChart(data: data)

        // Step 5: Analyze
        analyzePrecipitation(tempForecast: tempForecast, precipForecast: precipForecast, data: data)
        buildWeatherServiceEvents(data: data)
        computeAccuracy(tempForecast: tempForecast, data: data)
        buildSummary(tempForecast: tempForecast, data: data)

        updateElapsed()
        statusMessage = "Forecast complete for \(locationName)"
        isLoading = false
    }

    // MARK: - Temperature Chart

    private func buildTemperatureChart(data: WeatherData, tempForecast: [Float]) {
        var points = [ChartPoint]()
        let showHistorical = min(48, data.historicalCount)
        let histStart = data.historicalCount - showHistorical

        for i in 0 ..< showHistorical {
            let idx = histStart + i
            let date = idx < data.timestamps.count ? data.timestamps[idx] : nil
            points.append(
                ChartPoint(
                    id: points.count, hour: i, date: date,
                    temperature: data.temperature[idx], kind: .historical))
        }

        for i in 0 ..< tempForecast.count {
            let idx = data.historicalCount + i
            let date = idx < data.timestamps.count ? data.timestamps[idx] : nil
            points.append(
                ChartPoint(
                    id: points.count, hour: showHistorical + i, date: date,
                    temperature: tempForecast[i], kind: .totoForecast))
        }

        let actualCount = min(tempForecast.count, data.actualForecastCount)
        for i in 0 ..< actualCount {
            let idx = data.historicalCount + i
            let date = idx < data.timestamps.count ? data.timestamps[idx] : nil
            points.append(
                ChartPoint(
                    id: points.count, hour: showHistorical + i, date: date,
                    temperature: data.temperature[idx], kind: .actual))
        }

        temperaturePoints = points
    }

    // MARK: - Precipitation Chart

    private func buildPrecipChart(data: WeatherData, precipForecast: [Float]) {
        var points = [ChartPoint]()
        let showHistorical = min(48, data.historicalCount)
        let histStart = data.historicalCount - showHistorical

        for i in 0 ..< showHistorical {
            let idx = histStart + i
            let date = idx < data.timestamps.count ? data.timestamps[idx] : nil
            points.append(
                ChartPoint(
                    id: points.count, hour: i, date: date,
                    temperature: data.precipitation[idx], kind: .historical))
        }

        for i in 0 ..< precipForecast.count {
            let idx = data.historicalCount + i
            let date = idx < data.timestamps.count ? data.timestamps[idx] : nil
            points.append(
                ChartPoint(
                    id: points.count, hour: showHistorical + i, date: date,
                    temperature: precipForecast[i], kind: .totoForecast))
        }

        let actualCount = min(precipForecast.count, data.actualForecastCount)
        for i in 0 ..< actualCount {
            let idx = data.historicalCount + i
            let date = idx < data.timestamps.count ? data.timestamps[idx] : nil
            points.append(
                ChartPoint(
                    id: points.count, hour: showHistorical + i, date: date,
                    temperature: data.precipitation[idx], kind: .actual))
        }

        precipPoints = points
    }

    // MARK: - Snowfall Chart (cumulative, Weather Service only)

    private func buildSnowfallChart(data: WeatherData) {
        var points = [ChartPoint]()
        let showHistorical = min(48, data.historicalCount)
        let histStart = data.historicalCount - showHistorical

        // Cumulative snowfall for historical period
        var cumSnow: Float = 0
        for i in 0 ..< showHistorical {
            let idx = histStart + i
            let date = idx < data.timestamps.count ? data.timestamps[idx] : nil
            cumSnow += data.snowfall[idx]
            points.append(
                ChartPoint(
                    id: points.count, hour: i, date: date,
                    temperature: cumSnow, kind: .historical))
        }

        // Cumulative snowfall for forecast period
        let forecastCount = min(72, data.actualForecastCount)
        for i in 0 ..< forecastCount {
            let idx = data.historicalCount + i
            let date = idx < data.timestamps.count ? data.timestamps[idx] : nil
            cumSnow += data.snowfall[idx]
            points.append(
                ChartPoint(
                    id: points.count, hour: showHistorical + i, date: date,
                    temperature: cumSnow, kind: .actual))
        }

        // Only show chart if there's any snow
        let hasSnow = cumSnow > 0.1
        snowfallPoints = hasSnow ? points : []
    }

    // MARK: - Precipitation Analysis

    private func analyzePrecipitation(
        tempForecast: [Float], precipForecast: [Float], data: WeatherData
    ) {
        var events = [PrecipEvent]()
        var totalRain: Float = 0
        var totalSnow: Float = 0

        var inEvent = false
        var eventStart = 0
        var eventRain: Float = 0
        var eventSnow: Float = 0

        for i in 0 ..< precipForecast.count {
            let precip = precipForecast[i]
            let temp = tempForecast[i]
            let isSnow = temp <= 33
            let hasPrecip = precip > 0.005

            if hasPrecip && !inEvent {
                inEvent = true
                eventStart = i
                eventRain = 0
                eventSnow = 0
            }

            if hasPrecip {
                if isSnow {
                    eventSnow += precip * 10
                    totalSnow += precip * 10
                } else {
                    eventRain += precip
                    totalRain += precip
                }
            }

            if (!hasPrecip && inEvent) || (i == precipForecast.count - 1 && inEvent) {
                let baseIdx = data.historicalCount
                let startDate =
                    (baseIdx + eventStart) < data.timestamps.count
                    ? data.timestamps[baseIdx + eventStart] : nil
                let endDate =
                    (baseIdx + i) < data.timestamps.count
                    ? data.timestamps[baseIdx + i] : nil

                let type: String
                if eventSnow > 0 && eventRain > 0 { type = "Mix" }
                else if eventSnow > 0 { type = "Snow" }
                else { type = "Rain" }

                events.append(
                    PrecipEvent(
                        type: type, startHour: eventStart, endHour: i,
                        totalInches: eventRain + eventSnow,
                        startDate: startDate, endDate: endDate))
                inEvent = false
            }
        }

        self.precipEvents = events
        self.totalRainInches = totalRain > 0.01 ? totalRain : nil
        self.totalSnowInches = totalSnow > 0.1 ? totalSnow : nil
    }

    // MARK: - Weather Service Events

    private func buildWeatherServiceEvents(data: WeatherData) {
        var events = [PrecipEvent]()
        let forecastCount = min(72, data.actualForecastCount)
        let baseIdx = data.historicalCount

        var inEvent = false
        var eventStart = 0
        var eventRain: Float = 0
        var eventSnow: Float = 0

        for i in 0 ..< forecastCount {
            let idx = baseIdx + i
            guard idx < data.snowfall.count && idx < data.rain.count else { continue }
            let snow = data.snowfall[idx]
            let rain = data.rain[idx]
            let hasPrecip = snow > 0.01 || rain > 0.005

            if hasPrecip && !inEvent {
                inEvent = true
                eventStart = i
                eventRain = 0
                eventSnow = 0
            }

            if hasPrecip {
                eventSnow += snow
                eventRain += rain
            }

            if (!hasPrecip && inEvent) || (i == forecastCount - 1 && inEvent) {
                let startDate =
                    (baseIdx + eventStart) < data.timestamps.count
                    ? data.timestamps[baseIdx + eventStart] : nil
                let endDate =
                    (baseIdx + i) < data.timestamps.count
                    ? data.timestamps[baseIdx + i] : nil

                let type: String
                if eventSnow > 0.1 && eventRain > 0.01 { type = "Mix" }
                else if eventSnow > 0.1 { type = "Snow" }
                else { type = "Rain" }

                events.append(
                    PrecipEvent(
                        type: type, startHour: eventStart, endHour: i,
                        totalInches: type == "Rain" ? eventRain : eventSnow,
                        startDate: startDate, endDate: endDate))
                inEvent = false
            }
        }

        weatherServiceEvents = events
    }

    // MARK: - Accuracy

    private func computeAccuracy(tempForecast: [Float], data: WeatherData) {
        let overlapCount = min(tempForecast.count, data.actualForecastCount)
        guard overlapCount > 0 else {
            temperatureMAE = nil
            return
        }

        var totalError: Float = 0
        for i in 0 ..< overlapCount {
            let actual = data.temperature[data.historicalCount + i]
            let predicted = tempForecast[i]
            totalError += abs(actual - predicted)
        }
        temperatureMAE = totalError / Float(overlapCount)
    }

    // MARK: - Summary

    /// Weather service snowfall total (from GFS model) for display.
    var weatherServiceSnow: Float?

    private func buildSummary(tempForecast: [Float], data: WeatherData) {
        forecastHighTemp = tempForecast.max()
        forecastLowTemp = tempForecast.min()

        // Sum weather service snowfall over the forecast period
        let forecastCount = min(72, data.actualForecastCount)
        var wsSnow: Float = 0
        for i in 0 ..< forecastCount {
            let idx = data.historicalCount + i
            if idx < data.snowfall.count {
                wsSnow += data.snowfall[idx]
            }
        }
        weatherServiceSnow = wsSnow > 0.1 ? wsSnow : nil
    }

    // MARK: - Helpers

    private func updateElapsed() {
        guard let startTime else { return }
        let elapsed = Date().timeIntervalSince(startTime)
        if elapsed < 60 {
            elapsedTime = String(format: "%.1fs", elapsed)
        } else {
            let mins = Int(elapsed) / 60
            let secs = Int(elapsed) % 60
            elapsedTime = "\(mins)m \(secs)s"
        }
    }
}
