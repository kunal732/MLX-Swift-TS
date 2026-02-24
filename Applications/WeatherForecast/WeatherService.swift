import CoreLocation
import Foundation

/// Weather data fetched from Open-Meteo for a specific location.
struct WeatherData: Sendable {
    let locationName: String
    let latitude: Double
    let longitude: Double
    let timestamps: [Date]
    let temperature: [Float]       // °F
    let precipitation: [Float]     // inches
    let rain: [Float]              // inches
    let snowfall: [Float]          // inches
    let humidity: [Float]          // %
    let windSpeed: [Float]         // mph
    let pressure: [Float]          // hPa
    let cloudCover: [Float]        // %
    let dewPoint: [Float]          // °F

    /// Number of historical hours (past data used as model context).
    let historicalCount: Int
    /// Number of actual forecast hours from the weather service (for comparison).
    let actualForecastCount: Int
}

/// Fetches real weather data using Apple's geocoder and the Open-Meteo API.
enum WeatherService {

    // MARK: - Geocoding

    /// Convert a zip code to coordinates and location name.
    static func geocode(zipCode: String) async throws -> (
        name: String, latitude: Double, longitude: Double
    ) {
        let geocoder = CLGeocoder()
        let placemarks = try await geocoder.geocodeAddressString(zipCode)
        guard let placemark = placemarks.first,
            let location = placemark.location
        else {
            throw WeatherError.geocodingFailed("No results for zip code \(zipCode)")
        }
        let name = [placemark.locality, placemark.administrativeArea]
            .compactMap { $0 }
            .joined(separator: ", ")
        return (
            name: name.isEmpty ? zipCode : name,
            latitude: location.coordinate.latitude,
            longitude: location.coordinate.longitude
        )
    }

    // MARK: - Weather Data

    /// Fetch 7 days of historical hourly weather + 1 day of actual forecast.
    static func fetchWeather(latitude: Double, longitude: Double) async throws -> OpenMeteoResponse
    {
        var components = URLComponents(string: "https://api.open-meteo.com/v1/forecast")!
        components.queryItems = [
            URLQueryItem(name: "latitude", value: String(latitude)),
            URLQueryItem(name: "longitude", value: String(longitude)),
            URLQueryItem(
                name: "hourly",
                value:
                    "temperature_2m,precipitation,rain,snowfall,relative_humidity_2m,wind_speed_10m,surface_pressure,cloud_cover,dew_point_2m"),
            URLQueryItem(name: "past_days", value: "7"),
            URLQueryItem(name: "forecast_days", value: "4"),
            URLQueryItem(name: "temperature_unit", value: "fahrenheit"),
            URLQueryItem(name: "wind_speed_unit", value: "mph"),
            URLQueryItem(name: "precipitation_unit", value: "inch"),
            URLQueryItem(name: "timezone", value: "auto"),
        ]

        let (data, response) = try await URLSession.shared.data(from: components.url!)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw WeatherError.apiFailed("Open-Meteo returned non-200 status")
        }
        return try JSONDecoder().decode(OpenMeteoResponse.self, from: data)
    }

    /// Full pipeline: zip code -> geocode -> fetch weather -> structured data.
    static func getWeatherData(zipCode: String) async throws -> WeatherData {
        let geo = try await geocode(zipCode: zipCode)
        let response = try await fetchWeather(latitude: geo.latitude, longitude: geo.longitude)

        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm"
        dateFormatter.timeZone = TimeZone(identifier: response.timezone)

        let timestamps = response.hourly.time.compactMap { dateFormatter.date(from: $0) }

        // Past 7 days = 168 hours of history, rest is actual forecast
        let historicalCount = 7 * 24

        return WeatherData(
            locationName: geo.name,
            latitude: geo.latitude,
            longitude: geo.longitude,
            timestamps: timestamps,
            temperature: response.hourly.temperature_2m,
            precipitation: response.hourly.precipitation,
            rain: response.hourly.rain,
            snowfall: response.hourly.snowfall,
            humidity: response.hourly.relative_humidity_2m,
            windSpeed: response.hourly.wind_speed_10m,
            pressure: response.hourly.surface_pressure,
            cloudCover: response.hourly.cloud_cover,
            dewPoint: response.hourly.dew_point_2m,
            historicalCount: historicalCount,
            actualForecastCount: timestamps.count - historicalCount
        )
    }
}

// MARK: - API Response Models

struct OpenMeteoResponse: Decodable {
    let latitude: Double
    let longitude: Double
    let timezone: String
    let hourly: HourlyData

    struct HourlyData: Decodable {
        let time: [String]
        let temperature_2m: [Float]
        let precipitation: [Float]
        let rain: [Float]
        let snowfall: [Float]
        let relative_humidity_2m: [Float]
        let wind_speed_10m: [Float]
        let surface_pressure: [Float]
        let cloud_cover: [Float]
        let dew_point_2m: [Float]
    }
}

// MARK: - Errors

enum WeatherError: LocalizedError {
    case geocodingFailed(String)
    case apiFailed(String)

    var errorDescription: String? {
        switch self {
        case .geocodingFailed(let msg): return msg
        case .apiFailed(let msg): return msg
        }
    }
}
