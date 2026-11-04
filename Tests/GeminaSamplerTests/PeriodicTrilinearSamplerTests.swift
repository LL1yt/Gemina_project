import Foundation
import MLX
import XCTest
@testable import GeminaSampler

final class PeriodicTrilinearSamplerTests: XCTestCase {
    private typealias Sampler = PeriodicTrilinearSampler
    private var maxAbsoluteError = 0.0
    private var maxNormalizedError = 0.0

    override func setUp() {
        super.setUp()
        XCTAssertEqual(Device.defaultDevice().deviceType, .gpu, "M1 validation must run on Metal GPU")
    }

    override func tearDown() {
        print("M1 \(name): max_abs=\(maxAbsoluteError), max_normalized=\(maxNormalizedError)")
        super.tearDown()
    }

    private func close(
        _ actual: [Float], _ expected: [Double],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (a, e) in zip(actual, expected) {
            let error = abs(Double(a) - e)
            let normalized = error / (1e-5 + 1e-4 * max(abs(Double(a)), abs(e)))
            maxAbsoluteError = max(maxAbsoluteError, error)
            maxNormalizedError = max(maxNormalizedError, normalized)
            XCTAssertTrue(a.isFinite && e.isFinite, file: file, line: line)
            if !normalized.isFinite || normalized > 1 {
                XCTFail("actual=\(a), expected=\(e), normalized error=\(normalized)",
                        file: file, line: line)
                return
            }
        }
    }

    private func field(_ value: (Int, Int, Int, Int) -> Float) -> [Float] {
        var result = [Float]()
        for x in 0..<16 {
            for y in 0..<16 {
                for z in 0..<16 {
                    for c in 0..<32 { result.append(value(x, y, z, c)) }
                }
            }
        }
        return result
    }

    private func axisField() -> [Float] {
        field { x, y, z, c in
            switch c {
            case 0: Float(x) / 16
            case 1: Float(y) / 16
            case 2: Float(z) / 16
            default: Float(c) / 32
            }
        }
    }

    private func varyingField() -> [Float] {
        field { x, y, z, c in Float((x * 73 + y * 37 + z * 19 + c * 13) % 257 - 128) / 128 }
    }

    private func sample(_ w: [Float], _ rows: [[Float]]) throws -> Sampler.Sample {
        try Sampler.sample(
            field: MLXArray(w, [4096, 32]),
            raw: MLXArray(rows.flatMap { $0 }, [rows.count, 3])
        )
    }

    // Scalar Double oracle: no MLX, no production helper; nested logical axes
    // and positive remainder on Double avoid cast-before-wrap even for huge u.
    private func reference(_ w: [Float], _ u: [Float]) -> (read: [Double], ids: [Int32], weights: [Double]) {
        let whole = u.map { floor(Double($0)) }
        let fraction = zip(u, whole).map { Double($0.0) - $0.1 }
        let base = whole.map { j -> Int in
            let remainder = j.truncatingRemainder(dividingBy: 16)
            return Int(remainder < 0 ? remainder + 16 : remainder)
        }
        var result = [Double](repeating: 0, count: 32)
        var ids = [Int32]()
        var weights = [Double]()
        for dx in 0...1 {
            for dy in 0...1 {
                for dz in 0...1 {
                    let x = (base[0] + dx) % 16
                    let y = (base[1] + dy) % 16
                    let z = (base[2] + dz) % 16
                    let id = x * 256 + y * 16 + z
                    let wx = dx == 0 ? 1 - fraction[0] : fraction[0]
                    let wy = dy == 0 ? 1 - fraction[1] : fraction[1]
                    let wz = dz == 0 ? 1 - fraction[2] : fraction[2]
                    let weight = wx * wy * wz
                    ids.append(Int32(id))
                    weights.append(weight)
                    for c in 0..<32 { result[c] += weight * Double(w[id * 32 + c]) }
                }
            }
        }
        return (result, ids, weights)
    }

    private func check(_ result: Sampler.Sample, w: [Float], rows: [[Float]]) {
        XCTAssertEqual(result.read.shape, [rows.count, 32])
        XCTAssertEqual(result.position.shape, [rows.count, 3])
        XCTAssertEqual(result.indices.shape, [rows.count, 8])
        XCTAssertEqual(result.lambda.shape, [rows.count, 8])
        XCTAssertEqual(result.indices.dtype, .int32)
        for array in [result.read, result.position, result.lambda] {
            XCTAssertEqual(array.dtype, .float32)
        }
        let reads = result.read.asArray(Float.self)
        let ids = result.indices.asArray(Int32.self)
        let weights = result.lambda.asArray(Float.self)
        XCTAssertTrue(ids.allSatisfy { (0..<4096).contains($0) })
        XCTAssertTrue(weights.allSatisfy { $0.isFinite && $0 >= 0 })
        XCTAssertTrue(result.position.asArray(Float.self).allSatisfy { $0 >= 0 && $0 < 16 })
        for (b, row) in rows.enumerated() {
            let expected = reference(w, row)
            XCTAssertEqual(Array(ids[(b * 8)..<(b * 8 + 8)]), expected.ids)
            close(Array(weights[(b * 8)..<(b * 8 + 8)]), expected.weights)
            close([weights[(b * 8)..<(b * 8 + 8)].reduce(0, +)], [1])
            close(Array(reads[(b * 32)..<(b * 32 + 32)]), expected.read)
        }
    }

    func testS01LogicalAxesCornerOrderAndAllNodes() throws {
        let w = axisField()
        let single = try sample(w, [[2, 3, 4]])
        XCTAssertEqual(single.indices.asArray(Int32.self), [564, 565, 580, 581, 820, 821, 836, 837])
        XCTAssertEqual(single.lambda.asArray(Float.self), [1, 0, 0, 0, 0, 0, 0, 0])
        check(single, w: w, rows: [[2, 3, 4]])
        var rows = [[Float]]()
        for x in 0..<16 {
            for y in 0..<16 {
                for z in 0..<16 { rows.append([Float(x), Float(y), Float(z)]) }
            }
        }
        let allNodes = try sample(w, rows)
        XCTAssertEqual(allNodes.read.asArray(Float.self), w)
        check(allNodes, w: w, rows: rows)
    }

    func testS02WeightsMidpointAndSeededInteriors() throws {
        let w = varyingField()
        let hand = try sample(w, [[2.25, 3.5, 4.75]])
        close(hand.lambda.asArray(Float.self), [3, 9, 3, 9, 1, 3, 1, 3].map { Double($0) / 32 })
        let middle = try sample(w, [[2.5, 3.5, 4.5]])
        XCTAssertEqual(middle.lambda.asArray(Float.self), [Float](repeating: 0.125, count: 8))
        check(hand, w: w, rows: [[2.25, 3.5, 4.75]])
        check(middle, w: w, rows: [[2.5, 3.5, 4.5]])
        var state: UInt64 = 11
        var rows = [[Float]]()
        for _ in 0..<32 {
            var row = [Float]()
            for _ in 0..<3 {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                let cell = Int((state >> 32) % 16)
                let fraction = Float((state >> 40) % 1023 + 1) / 1024
                row.append(Float(cell) + fraction)
            }
            rows.append(row)
        }
        check(try sample(w, rows), w: w, rows: rows)
    }

    func testS03NegativeWrapPeriodsAndLargeCoordinates() throws {
        let w = varyingField()
        let row: [Float] = [-0.25, 16.5, -16.75]
        let origin = try sample(w, [row])
        XCTAssertEqual(origin.indices.asArray(Int32.self), [3855, 3840, 3871, 3856, 15, 0, 31, 16])
        close(origin.position.asArray(Float.self), [15.75, 0.5, 15.25])
        var rows = [row]
        for axis in 0..<3 {
            for k in [-1024, -3, -1, 1, 5, 1024] {
                var shifted = row
                shifted[axis] += Float(16 * k)
                rows.append(shifted)
            }
        }
        let periodic = try sample(w, rows)
        check(periodic, w: w, rows: rows)
        let reads = periodic.read.asArray(Float.self)
        for b in rows.indices {
            close(Array(reads[(32 * b)..<(32 * b + 32)]), origin.read.asArray(Float.self).map(Double.init))
        }
        let huge: Float = 1_099_511_627_776 // 2^40: cannot cast to Int32 before wrap
        let largeRows: [[Float]] = [[huge, -huge, huge], [.greatestFiniteMagnitude, -.greatestFiniteMagnitude, 0]]
        let large = try sample(w, largeRows)
        check(large, w: w, rows: largeRows)
        close(large.read.asArray(Float.self), Array(w[0..<32]).map(Double.init) + Array(w[0..<32]).map(Double.init))
        XCTAssertEqual(large.position.asArray(Float.self), [Float](repeating: 0, count: 6))
    }

    func testS04ConstantsAndLocalChannelRamps() throws {
        let rows: [[Float]] = [[2.25, 3.5, 4.75], [-0.125, 15.875, 16.25], [0, 0, 0]]
        let constants = field { _, _, _, c in Float(c - 16) / 32 }
        let result = try sample(constants, rows)
        check(result, w: constants, rows: rows)
        let expected = (0..<32).map { Double($0 - 16) / 32 }
        close(result.read.asArray(Float.self), rows.flatMap { _ in expected })
        let w = axisField()
        let ramp = try sample(w, [rows[0]])
        let expectedAxes: [Double] = [2.25 / 16, 3.5 / 16, 4.75 / 16]
        let expectedChannels = (3..<32).map { Double($0) / 32 }
        close(ramp.read.asArray(Float.self), expectedAxes + expectedChannels)
        check(ramp, w: w, rows: [rows[0]])
    }

    func testS05BatchIndependenceAndFP32Seams() throws {
        let w = varyingField()
        let rows: [[Float]] = [[2.25, 3.5, 4.75], [-0.25, 16.5, -16.75], [2.25, 3.5, 4.75]]
        let batch = try sample(w, rows)
        check(batch, w: w, rows: rows)
        for b in rows.indices {
            let single = try sample(w, [rows[b]])
            close(batch.read[b].asArray(Float.self), single.read.asArray(Float.self).map(Double.init))
            XCTAssertEqual(batch.indices[b].asArray(Int32.self), single.indices.asArray(Int32.self))
            close(batch.lambda[b].asArray(Float.self), single.lambda.asArray(Float.self).map(Double.init))
        }
        let permuted = try sample(w, [rows[1], rows[2], rows[0]])
        for (b, source) in [1, 2, 0].enumerated() {
            close(permuted.read[b].asArray(Float.self), batch.read[source].asArray(Float.self).map(Double.init))
        }
        var changed = rows
        changed[1] = [9.1, 8.2, 7.3]
        let modified = try sample(w, changed)
        for b in [0, 2] {
            XCTAssertEqual(modified.read[b].asArray(Float.self), batch.read[b].asArray(Float.self))
        }
        let epsilon: Float = 1.0 / 1_048_576 // 2^-20
        let tiny: Float = 1.0 / 33_554_432 // 2^-25: Float(1-tiny) rounds to 1
        let boundary: [Float] = [-epsilon, 0, epsilon, Float(16).nextDown, 16, Float(16).nextUp, -tiny]
        var seamRows = [[Float]]()
        for axis in 0..<3 {
            for value in boundary {
                var row: [Float] = [2.25, 3.5, 4.75]
                row[axis] = value
                seamRows.append(row)
            }
        }
        seamRows += [[-epsilon, -epsilon, -epsilon], [-tiny, -tiny, -tiny], [16, 16, 16]]
        check(try sample(w, seamRows), w: w, rows: seamRows)
        let endpoint = try sample(w, [[-tiny, -tiny, -tiny]])
        XCTAssertEqual(endpoint.position.asArray(Float.self), [0, 0, 0])
        XCTAssertEqual(endpoint.lambda.asArray(Float.self), [0, 0, 0, 0, 0, 0, 0, 1])
        // Known piecewise seam values for the x/16 channel, not a global ramp.
        let ramp = try sample(axisField(), [[-epsilon, 0, 0], [0, 0, 0], [epsilon, 0, 0]])
        close([ramp.read[0, 0].item(Float.self), ramp.read[1, 0].item(Float.self), ramp.read[2, 0].item(Float.self)],
              [15.0 / 16 * Double(epsilon), 0, Double(epsilon) / 16])
    }

    func testRejectsInvalidInputsBeforeGather() throws {
        let w = MLXArray(varyingField(), [4096, 32])
        let u = MLXArray([Float(0), 0, 0], [1, 3])
        func rejects(_ field: MLXArray, _ raw: MLXArray, _ expected: Sampler.InputError) {
            XCTAssertThrowsError(try Sampler.sample(field: field, raw: raw)) {
                XCTAssertEqual($0 as? Sampler.InputError, expected)
            }
        }
        rejects(w.reshaped([16, 16, 16, 32]), u, .fieldShape)
        rejects(w, u.reshaped([3]), .addressShape)
        rejects(w, MLXArray([Float](), [0, 3]), .addressShape)
        rejects(w, MLXArray([Float(0), 0, 0, 0], [1, 4]), .addressShape)
        rejects(w.asType(.float16), u, .dtype)
        rejects(w, u.asType(.int32), .dtype)
        for invalid in [Float.nan, Float.infinity, -Float.infinity] {
            rejects(w, MLXArray([invalid, 0, 0], [1, 3]), .nonFiniteInput)
            var invalidW = varyingField()
            invalidW[0] = invalid
            rejects(MLXArray(invalidW, [4096, 32]), u, .nonFiniteInput)
        }
    }
}
