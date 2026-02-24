import Foundation
import MLX
import MLXTimeSeries

/// Sample hourly temperature data (in °F) for 7 days = 168 hours.
///
/// This simulates a realistic daily temperature cycle with some noise.
enum SampleWeatherData {

    /// Generate synthetic hourly temperatures for the given number of days.
    ///
    /// Uses a sinusoidal daily cycle: base 55°F, amplitude 15°F, peak at 3 PM.
    /// - Parameter days: Number of days of data to generate.
    /// - Returns: Array of hourly temperature values.
    static func hourlyTemperatures(days: Int = 7) -> [Float] {
        var temps = [Float]()
        for hour in 0 ..< (days * 24) {
            let dayFraction = Float(hour % 24) / 24.0
            // Peak at hour 15 (3 PM), trough at hour 3 (3 AM)
            let dailyCycle = sin((dayFraction - 0.25) * 2.0 * .pi)
            // Add slight upward trend + noise
            let trend = Float(hour) * 0.01
            let noise = Float.random(in: -2.0...2.0)
            let temp = 55.0 + 15.0 * dailyCycle + trend + noise
            temps.append(temp)
        }
        return temps
    }

    /// Create a `TimeSeriesInput` from sample weather data, padded to patch alignment.
    ///
    /// - Parameter patchSize: Patch size for alignment (default 64).
    /// - Returns: A padded `TimeSeriesInput` ready for the model.
    static func makeInput(patchSize: Int = 64) -> TimeSeriesInput {
        let temps = hourlyTemperatures()
        let input = TimeSeriesInput.univariate(temps)
        return input.padded(toPatchSize: patchSize)
    }
}
