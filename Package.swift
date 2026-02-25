// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "mlxtoto",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "MLXTimeSeries",
            targets: ["MLXTimeSeries"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            .upToNextMinor(from: "0.30.6")
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMinor(from: "1.1.6")
        ),
    ],
    targets: [
        .target(
            name: "MLXTimeSeries",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Libraries/MLXTimeSeries",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "MLXTimeSeriesTests",
            dependencies: [
                "MLXTimeSeries",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Tests/MLXTimeSeriesTests",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)
