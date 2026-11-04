import MLX

/// The fixed M0 field: a 16³ torus with 32 FP32 channels per node.
public enum PeriodicTrilinearSampler {
    public enum InputError: Error, Equatable {
        case fieldShape
        case addressShape
        case dtype
        case nonFiniteInput
        case invalidIndex
    }

    public struct Sample {
        public let read: MLXArray       // [B,32], Float32
        public let position: MLXArray   // [B,3], Float32, diagnostic only
        public let indices: MLXArray    // [B,8], Int32
        public let lambda: MLXArray     // [B,8], Float32
    }

    /// Stateless read from W:[4096,32] at raw u:[B,3]. B must be positive.
    /// Validations synchronize detached checks before gathering. The read uses
    /// raw u, never the rounded diagnostic position. No parameters are mutated.
    public static func sample(field: MLXArray, raw: MLXArray) throws -> Sample {
        guard field.shape == [4096, 32] else { throw InputError.fieldShape }
        guard raw.ndim == 2, raw.shape[0] > 0, raw.shape[1] == 3 else {
            throw InputError.addressShape
        }
        guard field.dtype == .float32, raw.dtype == .float32 else {
            throw InputError.dtype
        }
        guard all(isFinite(stopGradient(raw))).item(Bool.self),
            all(isFinite(stopGradient(field))).item(Bool.self)
        else { throw InputError.nonFiniteInput }

        let j = stopGradient(floor(raw))
        let f = raw - j
        // Reduce in floating point BEFORE Int32 conversion: no signed remainder
        // and no overflowing cast for finite, very large raw coordinates.
        let base = j - 16 * floor(j / 16)
        guard all(base .>= 0).item(Bool.self), all(base .< 16).item(Bool.self) else {
            throw InputError.invalidIndex
        }
        let offsets = MLXArray([
            Int32(0), 0, 0,  0, 0, 1,  0, 1, 0,  0, 1, 1,
            1, 0, 0,  1, 0, 1,  1, 1, 0,  1, 1, 1,
        ], [1, 8, 3])
        let neighbors = base.asType(.int32).expandedDimensions(axis: 1) + offsets
        let q = which(neighbors .>= 16, neighbors - 16, neighbors)
        let indices = (q[.ellipsis, 0] * 16 + q[.ellipsis, 1]) * 16 + q[.ellipsis, 2]

        let fractions = f.expandedDimensions(axis: 1)
        let factors = which(offsets .== 0, 1 - fractions, fractions)
        let lambda = factors[.ellipsis, 0] * factors[.ellipsis, 1] * factors[.ellipsis, 2]
        let gathered = take(field, indices, axis: 0) // [B,8,32]
        let read = (gathered * lambda.expandedDimensions(axis: 2)).sum(axis: 1)

        // FP32 can round i+f to 16 at the seam; canonicalize this diagnostic
        // endpoint only. Detaching it does not detach f/lambda/read.
        let p = stopGradient(base + f)
        let position = which(p .>= 16, p - 16, p)
        return Sample(read: read, position: position, indices: indices, lambda: lambda)
    }
}
