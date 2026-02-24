import Foundation

/// Fixed-size circular buffer for time series samples.
struct RollingBuffer: Codable {
    private var storage: [Float]
    private var head: Int = 0
    private var isFull: Bool = false
    let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
        self.storage = [Float](repeating: 0, count: capacity)
    }

    /// Initialize from a saved array, restoring into a buffer of the given capacity.
    init(capacity: Int, restoring values: [Float]) {
        self.capacity = capacity
        self.storage = [Float](repeating: 0, count: capacity)
        for v in values { append(v) }
    }

    var count: Int { isFull ? capacity : head }

    mutating func append(_ value: Float) {
        storage[head] = value
        head = (head + 1) % capacity
        if head == 0 { isFull = true }
    }

    /// Return all samples in chronological order.
    func toArray() -> [Float] {
        if isFull {
            return Array(storage[head...]) + Array(storage[..<head])
        } else {
            return Array(storage[..<head])
        }
    }

    /// Last N samples (most recent).
    func last(_ n: Int) -> [Float] {
        let all = toArray()
        return Array(all.suffix(n))
    }

    var latest: Float? {
        count > 0 ? storage[(head - 1 + capacity) % capacity] : nil
    }
}
