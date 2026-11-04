# M1 sampler fixtures

Prepared by the implementing Codex agent before writing the sampler. These are
implementation-side analytical fixtures, not Virgil's independent audit.
Contract: PR #5 at `40811f3c4cc843972cccc348b2b6c68678cc4eb9`, sections 1/4/6
of Mathematical and Interface Contracts and S01–S05 of Test Matrix.
Base: `5a039b38f11e1d3e13c2b12151c22570996f1369`.

N=16, C=32, FP32; corner order 000,001,010,011,100,101,110,111.
Integer IDs, shapes and dtypes compare exactly. Numeric comparisons use the
unchanged contract bound `1e-5 + 1e-4 * max(abs(expected), abs(actual))`.
Tests report maximum absolute error and error divided by that bound (≤1 passes).

| ID | Inputs and analytical expectation | Detects |
|---|---|---|
| S01 | u=(2,3,4); indices [564,565,580,581,820,821,836,837]. An independently constructed logical field has channels x/16, y/16, z/16 and c/32. At integer u the read is the exact node; exhaust all 4096 base nodes. | Corner order, axis/stride/channel confusion, Int32 indices, FP32 outputs |
| S02 | u=(2.25,3.5,4.75); λ=[3,9,3,9,1,3,1,3]/32. At cell midpoint λ=1/8, at nodes only corner 000 has weight 1. Scalar Double reference on deterministic nonconstant W and interior points (LCG seed 11). | Weights, partition of unity, weighted gather/reduction, B=1 axis |
| S03 | u=(-0.25,16.5,-16.75); base=(15,0,15), f=(.75,.5,.25); indices [3855,3840,3871,3856,15,0,31,16]. Shift each axis separately by 16k (negative/positive multiple periods). Very large representable FP32 coordinates ±2^40 and ±Float.greatestFiniteMagnitude are multiples of 16 and sample origin. | Signed remainder, cast before wrap, out-of-range indices, periodicity |
| S04 | Channel-specific constant W_c=(c-16)/32 is invariant. Within cell (2,3,4), channels x/16,y/16,z/16 interpolate the corresponding coordinate; channel c≥3 is c/32. | Constant reproduction, channel mixing, local affine reproduction; no global periodic-ramp assumption |
| S05 | Batch containing distinct/repeated rows equals individual calls; permuting/changing one row leaves other rows unchanged. At each axis use -ε,0,+ε,16.nextDown,16,16.nextUp, with ε=2^-20, and tiny negative 2^-25 (f may round to 1). Canonical p remains [0,16); p rounded to 16 becomes 0. Seam values use a scalar reference from raw u, never a second read from rounded p. | Cross-row mixing, squeezed dimensions, wrap seam and FP32 endpoint rounding |

Additional input checks reject wrong shapes/dtypes and NaN/±Inf before indexing;
no clamping. Tests cover B=1 and B=32 as well as exhaustive node batching.
Finite FP32 inputs already lacking a fractional part cannot recover it; these
tests assert semantics of the represented input, not exact real arithmetic.

M2/G01–G04, gradients, GRU, training and performance benchmarks are outside this
task. No gradient PASS is implied by forward-value fixtures.
