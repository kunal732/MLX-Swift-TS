import Darwin
import Foundation

#if os(macOS)
import IOKit
#endif

#if os(iOS)
import UIKit
#endif

/// Snapshot of system metrics at a point in time.
struct MetricsSnapshot {
    var cpuPercent: Float = 0        // 0-100
    var memoryUsedMB: Float = 0
    var memoryTotalMB: Float = 0
    var memoryPercent: Float = 0     // 0-100
    var diskReadMBps: Float = 0
    var diskWriteMBps: Float = 0
    var netInMBps: Float = 0
    var netOutMBps: Float = 0
    var gpuPercent: Float = 0        // 0-100

    #if os(iOS)
    var thermalState: Float = 0      // 0-3 (nominal, fair, serious, critical)
    var batteryLevel: Float = 0      // 0-100
    var availableMemoryMB: Float = 0
    #endif
}

#if os(macOS)

/// Collects system metrics by polling macOS APIs.
class MetricsCollector {

    private var prevCPUTicks: (user: UInt64, system: UInt64, idle: UInt64, nice: UInt64)?
    private var prevDiskRead: Int64 = 0
    private var prevDiskWrite: Int64 = 0
    private var prevNetIn: UInt64 = 0
    private var prevNetOut: UInt64 = 0
    private var prevTime: Date?

    func collect() -> MetricsSnapshot {
        let now = Date()
        let dt = prevTime.map { Float(now.timeIntervalSince($0)) } ?? 1.0
        prevTime = now

        var snap = MetricsSnapshot()
        snap.cpuPercent = collectCPU(dt: dt)
        collectMemory(&snap)
        collectDiskIO(&snap, dt: dt)
        collectNetworkIO(&snap, dt: dt)
        snap.gpuPercent = collectGPU()
        return snap
    }

    // MARK: - CPU

    private func collectCPU(dt: Float) -> Float {
        var loadInfo = host_cpu_load_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<host_cpu_load_info_data_t>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &loadInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }

        let user = UInt64(loadInfo.cpu_ticks.0)
        let system = UInt64(loadInfo.cpu_ticks.1)
        let idle = UInt64(loadInfo.cpu_ticks.2)
        let nice = UInt64(loadInfo.cpu_ticks.3)

        defer { prevCPUTicks = (user, system, idle, nice) }

        guard let prev = prevCPUTicks else { return 0 }
        let dUser = Float(user - prev.user)
        let dSystem = Float(system - prev.system)
        let dIdle = Float(idle - prev.idle)
        let dNice = Float(nice - prev.nice)
        let total = dUser + dSystem + dIdle + dNice
        guard total > 0 else { return 0 }
        return (dUser + dSystem + dNice) / total * 100
    }

    // MARK: - Memory

    private func collectMemory(_ snap: inout MetricsSnapshot) {
        let totalBytes = ProcessInfo.processInfo.physicalMemory
        snap.memoryTotalMB = Float(totalBytes) / 1_048_576

        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return }

        let pageSize = UInt64(vm_kernel_page_size)
        let active = UInt64(stats.active_count) * pageSize
        let wired = UInt64(stats.wire_count) * pageSize
        let compressed = UInt64(stats.compressor_page_count) * pageSize
        let used = active + wired + compressed

        snap.memoryUsedMB = Float(used) / 1_048_576
        snap.memoryPercent = snap.memoryUsedMB / snap.memoryTotalMB * 100
    }

    // MARK: - Disk I/O

    private func collectDiskIO(_ snap: inout MetricsSnapshot, dt: Float) {
        var iterator: io_iterator_t = 0
        let matching = IOServiceMatching("IOBlockStorageDriver")
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS
        else { return }
        defer { IOObjectRelease(iterator) }

        var totalRead: Int64 = 0
        var totalWrite: Int64 = 0
        var entry = IOIteratorNext(iterator)
        while entry != 0 {
            defer { IOObjectRelease(entry); entry = IOIteratorNext(iterator) }
            var props: Unmanaged<CFMutableDictionary>?
            guard
                IORegistryEntryCreateCFProperties(entry, &props, kCFAllocatorDefault, 0)
                    == KERN_SUCCESS,
                let dict = props?.takeRetainedValue() as? [String: Any],
                let stats = dict["Statistics"] as? [String: Any]
            else { continue }
            totalRead += stats["Bytes (Read)"] as? Int64 ?? 0
            totalWrite += stats["Bytes (Write)"] as? Int64 ?? 0
        }

        if prevDiskRead > 0 && dt > 0 {
            snap.diskReadMBps = Float(totalRead - prevDiskRead) / 1_048_576 / dt
            snap.diskWriteMBps = Float(totalWrite - prevDiskWrite) / 1_048_576 / dt
        }
        prevDiskRead = totalRead
        prevDiskWrite = totalWrite
    }

    // MARK: - Network I/O

    private func collectNetworkIO(_ snap: inout MetricsSnapshot, dt: Float) {
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return }
        defer { freeifaddrs(ifaddr) }

        var totalIn: UInt64 = 0
        var totalOut: UInt64 = 0
        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let addr = cursor {
            defer { cursor = addr.pointee.ifa_next }
            guard addr.pointee.ifa_addr.pointee.sa_family == UInt8(AF_LINK) else { continue }
            let name = String(cString: addr.pointee.ifa_name)
            guard !name.hasPrefix("lo") else { continue }
            let data = addr.pointee.ifa_data.assumingMemoryBound(to: if_data.self).pointee
            totalIn += UInt64(data.ifi_ibytes)
            totalOut += UInt64(data.ifi_obytes)
        }

        if prevNetIn > 0 && dt > 0 && totalIn >= prevNetIn && totalOut >= prevNetOut {
            snap.netInMBps = Float(totalIn - prevNetIn) / 1_048_576 / dt
            snap.netOutMBps = Float(totalOut - prevNetOut) / 1_048_576 / dt
        }
        prevNetIn = totalIn
        prevNetOut = totalOut
    }

    // MARK: - GPU

    private func collectGPU() -> Float {
        var iterator: io_iterator_t = 0
        let matching = IOServiceMatching("IOAccelerator")
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS
        else { return 0 }
        defer { IOObjectRelease(iterator) }

        var entry = IOIteratorNext(iterator)
        while entry != 0 {
            defer { IOObjectRelease(entry); entry = IOIteratorNext(iterator) }
            var props: Unmanaged<CFMutableDictionary>?
            guard
                IORegistryEntryCreateCFProperties(entry, &props, kCFAllocatorDefault, 0)
                    == KERN_SUCCESS,
                let dict = props?.takeRetainedValue() as? [String: Any],
                let perf = dict["PerformanceStatistics"] as? [String: Any]
            else { continue }
            if let util = perf["Device Utilization %"] as? Int { return Float(util) }
            if let util = perf["GPU Activity(%)"] as? Int { return Float(util) }
        }
        return 0
    }
}

#elseif os(iOS)

/// Collects app-level metrics on iOS (no system-wide stats available).
class MetricsCollector {

    private var batteryMonitoringEnabled = false

    func collect() -> MetricsSnapshot {
        if !batteryMonitoringEnabled {
            UIDevice.current.isBatteryMonitoringEnabled = true
            batteryMonitoringEnabled = true
        }

        var snap = MetricsSnapshot()
        snap.cpuPercent = collectAppCPU()
        collectAppMemory(&snap)
        snap.thermalState = collectThermalState()
        snap.batteryLevel = collectBattery()
        return snap
    }

    // MARK: - App CPU (per-thread sum)

    private func collectAppCPU() -> Float {
        var threadList: thread_act_array_t?
        var threadCount: mach_msg_type_number_t = 0

        let result = task_threads(mach_task_self_, &threadList, &threadCount)
        guard result == KERN_SUCCESS, let threads = threadList else { return 0 }
        defer {
            vm_deallocate(
                mach_task_self_,
                vm_address_t(bitPattern: threads),
                vm_size_t(Int(threadCount) * MemoryLayout<thread_t>.stride))
        }

        var totalCPU: Float = 0
        for i in 0..<Int(threadCount) {
            var info = thread_basic_info_data_t()
            var infoCount = mach_msg_type_number_t(
                MemoryLayout<thread_basic_info_data_t>.size / MemoryLayout<integer_t>.size)
            let kr = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: Int(infoCount)) {
                    thread_info(threads[i], thread_flavor_t(THREAD_BASIC_INFO), $0, &infoCount)
                }
            }
            guard kr == KERN_SUCCESS else { continue }
            if info.flags & TH_FLAGS_IDLE != 0 { continue }
            totalCPU += Float(info.cpu_usage) / Float(TH_USAGE_SCALE) * 100
        }

        return totalCPU
    }

    // MARK: - App Memory

    private func collectAppMemory(_ snap: inout MetricsSnapshot) {
        var info = mach_task_basic_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kr == KERN_SUCCESS {
            snap.memoryUsedMB = Float(info.resident_size) / 1_048_576
        }

        snap.availableMemoryMB = Float(os_proc_available_memory()) / 1_048_576
        snap.memoryTotalMB = snap.memoryUsedMB + snap.availableMemoryMB
        if snap.memoryTotalMB > 0 {
            snap.memoryPercent = snap.memoryUsedMB / snap.memoryTotalMB * 100
        }
    }

    // MARK: - Thermal State

    private func collectThermalState() -> Float {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:  return 0
        case .fair:     return 1
        case .serious:  return 2
        case .critical: return 3
        @unknown default: return 0
        }
    }

    // MARK: - Battery

    private func collectBattery() -> Float {
        let level = UIDevice.current.batteryLevel
        return level >= 0 ? level * 100 : 0
    }
}

#endif
