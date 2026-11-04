// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "GeminaSampler",
    platforms: [.macOS(.v14)],
    products: [.library(name: "GeminaSampler", targets: ["GeminaSampler"])],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            exact: "0.33.0"
        )
    ],
    targets: [
        .target(
            name: "GeminaSampler",
            dependencies: [.product(name: "MLX", package: "mlx-swift")],
            path: "src/GeminaSampler"
        ),
        .testTarget(
            name: "GeminaSamplerTests",
            dependencies: ["GeminaSampler"]
        )
    ]
)
