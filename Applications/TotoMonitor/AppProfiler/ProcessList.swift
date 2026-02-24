import Darwin
import Foundation

struct ProcessInfo2 {
    var pid: pid_t
    var name: String
    var cpuPercent: Float
    var memoryMB: Float
}

/// Enumerates running processes and tracks per-process CPU/memory.
class ProcessListCollector {

    // Previous CPU times for delta calculation
    private var prevTimes: [pid_t: (user: UInt64, system: UInt64, wall: Date)] = [:]

    func collectAll() -> [ProcessInfo2] {
        let pids = allPIDs()
        var results = [ProcessInfo2]()
        let now = Date()

        for pid in pids {
            guard let info = infoForPID(pid, now: now) else { continue }
            results.append(info)
        }

        // Sort by CPU descending
        results.sort { $0.cpuPercent > $1.cpuPercent }
        return results
    }

    private func allPIDs() -> [pid_t] {
        let estimated = proc_listallpids(nil, 0)
        var pids = [pid_t](repeating: 0, count: Int(estimated) * 2)
        let bufSize = Int32(pids.count * MemoryLayout<pid_t>.size)
        let actual = proc_listallpids(&pids, bufSize)
        return Array(pids.prefix(Int(actual)))
    }

    private func infoForPID(_ pid: pid_t, now: Date) -> ProcessInfo2? {
        var info = proc_taskallinfo()
        let size = Int32(MemoryLayout<proc_taskallinfo>.size)
        guard proc_pidinfo(pid, PROC_PIDTASKALLINFO, 0, &info, size) == size else {
            return nil
        }

        let name = withUnsafePointer(to: info.pbsd.pbi_comm) {
            $0.withMemoryRebound(to: CChar.self, capacity: Int(MAXCOMLEN)) {
                String(cString: $0)
            }
        }

        let memMB = Float(info.ptinfo.pti_resident_size) / 1_048_576
        let userNs = info.ptinfo.pti_total_user
        let sysNs = info.ptinfo.pti_total_system

        // Calculate CPU% from delta
        var cpuPct: Float = 0
        if let prev = prevTimes[pid] {
            let dt = Float(now.timeIntervalSince(prev.wall))
            if dt > 0 {
                let dUser = Float(userNs - prev.user) / 1_000_000_000
                let dSys = Float(sysNs - prev.system) / 1_000_000_000
                cpuPct = (dUser + dSys) / dt * 100
            }
        }
        prevTimes[pid] = (userNs, sysNs, now)

        return ProcessInfo2(pid: pid, name: name, cpuPercent: cpuPct, memoryMB: memMB)
    }
}
