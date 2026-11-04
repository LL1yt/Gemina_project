# M1 periodic sampler — implementation and validation

2026-11-04. **M1 forward checks PASS on the local Metal GPU; independent audit pending.**
One implementing Codex agent, in the owner-authorized Morgana role. No second
implementer or reviewer was spawned. Virgil's fixtures/code were not consulted.
This report does not claim M2 gradients, a working Lion, training or acceptance into main.

## Revisions and sources

- Base main: `5a039b38f11e1d3e13c2b12151c22570996f1369` (rechecked before publication).
- Local tested code, tests, manifest and lockfile commit: `7110c25ea05a8202fea96885c8f6693e72e41405`.
  Later changes are documentation/evidence only. The public head is in PR metadata;
  GitHub API publication creates its own commit identity, with tree equality
  checked against the local deliverable. The local code commit is not claimed
  to be a separately published GitHub commit.
- Branch: `codex/m1-periodic-sampler`, isolated worktree backed by a separate local
  bare clone. The owner's original checkout and other worktrees/branches were not edited.
- [PR #5 contract](https://github.com/LL1yt/Gemina_project/blob/40811f3c4cc843972cccc348b2b6c68678cc4eb9/docs/plans/2026-11-02-mvp/MATHEMATICAL-INTERFACE-CONTRACTS.md)
  and [test matrix](https://github.com/LL1yt/Gemina_project/blob/40811f3c4cc843972cccc348b2b6c68678cc4eb9/docs/plans/2026-11-02-mvp/TEST-MATRIX-ACCEPTANCE.md),
  full source SHA `40811f3c4cc843972cccc348b2b6c68678cc4eb9`.
- Read current AGENTS, START-HERE, documentation map, PROVENANCE, CONTRIBUTING,
  [M0 passport](M0-IMPLEMENTATION-PASSPORT-v0.1.md), [MVP baseline](MVP-BASELINE-v0.1.md),
  role allocation, and PR #5 implementation/training/test-workflow contracts.
  Project_Index WORKING-PREFERENCES and NEW-PROJECT were read through GitHub;
  no changes to that repository. No `.agents/skills` or nested AGENTS/SKILL files
  existed in the checked Gemina base/worktree, so no local skill applied.
- The current owner instruction supersedes the old two-implementer proposal in
  PR #5. Neither that PR nor any other agent's branch was merged or rewritten.

## Environment actually used

| Component | Observed version/revision |
|---|---|
| Hardware | Apple M4 Max, MacBook Pro, arm64, 128 GB memory |
| macOS | 27.0.1, build 26A434 |
| Xcode | 27.0, build 27A266a; macOS SDK 27.0 |
| Swift | Apple Swift 6.4, swiftlang-6.4.0.34.1, clang-2100.3.34.1; driver 1.168.6 |
| Package manifest | Swift tools 6.3, deployment macOS 14, no `.xcodeproj` |
| MLX Swift | 0.33.0, `7ede4fd6d05f63305c1c0ebf4f8c1909a637c574` |
| MLX core gitlink | v0.32.4, `36969c094354086e15d2a327a9c50feb20b9cf11` |
| mlx-c gitlink | `a26f2d584a14ec619ae8db9a09be03becd47598a` |
| swift-numerics | 1.1.1, `0c0290ff6b24942dadb83a929ffaaa1481df04a2` |
| Backend | Metal GPU; every XCTest asserts `Device.defaultDevice().deviceType == .gpu` |

`swift package resolve` ran successfully. [Package.resolved](../Package.resolved)
records the pins; the MLX Swift SHA matches M0. Only the MLX product is linked.
No dependency version, architectural parameter, formula or numerical tolerance
was changed. Upstream [build instructions](https://github.com/ml-explore/mlx-swift/tree/7ede4fd6d05f63305c1c0ebf4f8c1909a637c574#xcodebuild)
were checked; tests use Xcode to build Metal shaders.

## Implementation

[PeriodicTrilinearSampler](../src/GeminaSampler/PeriodicTrilinearSampler.swift)
accepts FP32 W `[4096,32]` and raw u `[B,3]` with B>0, returning read `[B,32]`,
diagnostic canonical p `[B,3]`, Int32 indices `[B,8]` and FP32 lambda `[B,8]`.
It uses `j=stopGradient(floor(u))`, `f=u-j`, positive floating wrap before Int32
conversion, the contract's corner order/flatten rule, an axis-0 gather, and an
axis-1 weighted reduction. p is detached and never used to recompute the read.
Shape/dtype/nonfinite inputs and invalid base indices are rejected, not clamped.

The [analytical fixture design](M1-SAMPLER-FIXTURES.md) was written before the
sampler. [Tests](../Tests/GeminaSamplerTests/PeriodicTrilinearSamplerTests.swift)
combine hand-computed cases, all 4096 integer nodes, a scalar Double oracle and
32 deterministic interior rows (explicit LCG seed 11). No model/dataset RNG is used.

## Executed checks and results

The successful command (run 3, exit 0):

```sh
swift package resolve
xcodebuild test -scheme GeminaSampler \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath DerivedData -clonedSourcePackagesDirPath .build \
  -disableAutomaticPackageResolution -parallel-testing-enabled NO \
  -resultBundlePath /tmp/gemina-m1-tests-3.xcresult
```

Use a fresh result-bundle path when repeating. **6 XCTest cases, 0 failures,
0 skipped**, including S01–S05 and invalid-input checks. Tests completed in
2.494 seconds (not a performance benchmark).

| ID | Actual coverage | Max absolute error | Max normalized error | Result |
|---|---|---:|---:|---|
| S01 | Axes/channels, exact corner order/indices, all integer nodes, B=1 shapes/dtypes | 0 | 0 | PASS |
| S02 | Hand weights, midpoint, seeded interiors, nonnegative weights and sum, scalar weighted oracle | 1.38156e-7 | 0.00303554 | PASS |
| S03 | Negative wrap, per-axis ±multiple periods, indices, raw ±2^40 and ±largest finite FP32 | 0 | 0 | PASS |
| S04 | Per-channel constants and local affine ramps | 0 | 0 | PASS |
| S05 | Individual/batched reads, repeated/permuted rows, one-row perturbation, each axis at/near seams and rounded endpoint | 1.47847e-7 | 0.00317013 | PASS |
| Input guards | Wrong rank/shape, empty B, wrong dtype, NaN/±Inf in u and W | — | — | PASS |

Normalized error is `abs(a-e)/(1e-5+1e-4*max(abs(a),abs(e)))`; threshold ≤1.
This is the unchanged PR #5 bound. The two maxima need not come from the same element.

Evidence: [test output excerpt](validation/M1-TEST-OUTPUT.txt) and
[xcresult summary](validation/M1-XCRESULT-SUMMARY.json), exported from the actual
result bundle with device identifiers omitted. Full local log:
`/tmp/gemina-m1-tests-3.log`, SHA-256
`2723805ce2e7aab54a1c8b45858fb0a3847435e2a51ff38040a4f98d3ebeea6a`.
The full `.xcresult` remains local; its presence in GitHub is not claimed.

Earlier attempts are not hidden: an early Xcode scheme listing failed while
the test directory did not yet exist; run 1 failed to compile an invalid `.prod`
API name; run 2 failed the Swift type checker's expression-complexity limit in
a test expectation. These were corrected before run 3; neither run executed
the test suite. Upstream Metal compilation emitted C++17-extension/unused-constant
warnings; these did not fail the successful build or the tests.

Additional actual checks: `git diff --check`; relative links in changed Markdown;
tested source/manifest/lockfile bytes unchanged since the code commit; original
checkout still clean at `9bced371a2d4f836be15484b0100d1916435c3fa`; original DOCX SHA-256 matches
PROVENANCE (`525957abbd3a68ad1eeb9f83d0a8ab5a06dfc9a5b71bfebf39d58b1f10209f49`).

## Limits and next review

- **NOT_RUN:** M2/G01–G04 autodiff/finite differences, M3+, GRU, training,
  inference/generation, checkpoints, benchmarks, CPU parity, release configuration,
  other hardware/toolchains, CI and Virgil's independent audit.
- No current build/test blocker. Initial sandbox network and xcresult report-cache
  access restrictions were resolved through approved tool execution, without
  changing dependencies or bypassing a denial.
- CLI `git push` could not authenticate (`could not read Username for https://github.com`).
  It did not publish the branch. Publication uses the connected GitHub API instead;
  no credential extraction or authentication configuration changes are needed.
- Validation synchronizes detached checks, including a scan of W, on every call.
  No throughput claim is made. The API is an eager checked reference; compiled
  training integration is untested and outside M1.
- Very large FP32 values may already have lost fractional information. The sampler
  preserves the represented-input semantics; it cannot recover that information.
- This is implementation-side evidence. Independent acceptance remains with the
  owner/Virgil; no Agent League score or architecture advantage is claimed.
- Project_Index links were inspected; no inter-project technical dependency changed.
  Its overview was not edited. Publication of this Draft does not update main.
