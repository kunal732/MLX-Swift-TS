import Foundation
import IOKit
import IOKit.ps

struct BatterySnapshot {
    var percent: Float = 0       // 0-100
    var watts: Float = 0         // current power draw
    var isCharging: Bool = false
    var isPluggedIn: Bool = false
    var minutesRemaining: Int?   // nil if unknown/calculating
    var cycleCount: Int = 0
}

/// Collects battery information from IOKit.
class BatteryCollector {

    func collect() -> BatterySnapshot? {
        var snap = BatterySnapshot()

        // IOPowerSources for percent and charging state
        guard let psInfo = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
            let sources = IOPSCopyPowerSourcesList(psInfo)?.takeRetainedValue() as? [Any],
            let source = sources.first,
            let desc = IOPSGetPowerSourceDescription(psInfo, source as CFTypeRef)?
                .takeUnretainedValue() as? [String: Any]
        else { return nil }

        snap.percent = Float(desc[kIOPSCurrentCapacityKey] as? Int ?? 0)
        snap.isCharging = desc[kIOPSIsChargingKey] as? Bool ?? false
        let powerState = desc[kIOPSPowerSourceStateKey] as? String
        snap.isPluggedIn = (powerState == kIOPSACPowerValue)

        let tte = desc[kIOPSTimeToEmptyKey] as? Int ?? -1
        snap.minutesRemaining = tte >= 0 ? tte : nil

        // IORegistry for watts and cycle count
        let service = IOServiceGetMatchingService(
            kIOMainPortDefault, IOServiceNameMatching("AppleSmartBattery"))
        if service != 0 {
            defer { IOObjectRelease(service) }
            var props: Unmanaged<CFMutableDictionary>?
            if IORegistryEntryCreateCFProperties(service, &props, kCFAllocatorDefault, 0)
                == KERN_SUCCESS,
                let dict = props?.takeRetainedValue() as? [String: Any]
            {
                let voltage = Float(dict["Voltage"] as? Int ?? 0) / 1000.0
                let amperage = Float(dict["Amperage"] as? Int ?? 0) / 1000.0
                snap.watts = voltage * abs(amperage)
                snap.cycleCount = dict["CycleCount"] as? Int ?? 0
            }
        }

        return snap
    }
}
