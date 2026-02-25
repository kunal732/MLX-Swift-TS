# MLX-Swift-TS

Run Time series foundation models on Apple Silicon.

![Platform](https://img.shields.io/badge/platform-macOS%2014%2B%20%7C%20iOS%2017%2B-lightgrey)
![Swift](https://img.shields.io/badge/Swift-5.9%2B-orange)

## Overview

MLX-Swift-TS is a Swift SDK for running time series foundation models locally on Mac and iOS using [MLX](https://github.com/ml-explore/mlx-swift). Convert time series models hosted on HuggingFace Hub or stored locally to MLX format, then run inference - no server required.

Eight model architectures are supported out of the box, from Datadog's Toto to Google's TimesFM, Amazon's Chronos, and more.

## Installation

Add MLX-Swift-TS to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/kunal732/MLX-Swift-TS", branch: "main"),
]

// Add to your target
.product(name: "MLXTimeSeries", package: "MLX-Swift-TS")
```

## Quick Start

Time series forecasting predicts future values from a sequence of past measurements (CPU usage, heart rate, stock price, sensor readings). MLX-Swift-TS runs these models fully on-device using Apple Silicon.

### 1. Load a model

```swift
import MLXTimeSeries

// Load a pre-converted MLX time series model from HuggingFace Hub

let forecaster = try await TimeSeriesForecaster.loadFromHub(
    id: "kunal732/Toto-Open-Base-1.0-MLX"
)
```

### 2. Univariate: one variable over time

Use this when you have a single stream of measurements (e.g. CPU usage, temperature, sales).

```swift
// Historical CPU usage (%) - any length works, more history = better forecast

let cpuUsage: [Float] = [20, 22, 19, 23, 21, 25, 24, 26, 27, 25]

let input = TimeSeriesInput.univariate(cpuUsage)
let forecast = forecaster.forecast(input: input, predictionLength: 8)

print(forecast.mean)       // [1, 1, 8]    - predicted values for next 8 steps
print(forecast.quantiles)  // [1, 1, 8, Q] - uncertainty ranges (when available, see Reading the Output)
```

### 3. Multivariate: multiple variables over time

Use this when you have related signals measured together (e.g. heart rate + temperature). The model learns relationships between variables to improve all forecasts.

```swift
// Two variables, each with 6 historical readings

let series: [[Float]] = [
    [70, 72, 74, 73, 75, 76],              // Heart rate (bpm)
    [36.5, 36.6, 36.7, 36.6, 36.8, 36.9]  // Temperature (°C)
]

let input = TimeSeriesInput.multivariate(series)
let forecast = forecaster.forecast(input: input, predictionLength: 10)

print(forecast.mean)  // [1, 2, 10] - 10 predicted steps for each of the 2 variables
```

### Mental model

```
Univariate                    Multivariate
─────────────────────────     ─────────────────────────
Input:  [T]                   Input:  [N, T]
Output: [H]                   Output: [N, H]

T = historical steps          N = number of variables
H = prediction length         T = historical steps
                              H = prediction length
```

## Reading the Output

Every forecast returns a `TimeSeriesPrediction` with three fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `mean` | `[B, V, H]` | Point forecast - your best single predicted value per step |
| `quantiles` | `[B, V, H, Q]` | Uncertainty bands - the range of plausible outcomes (available on TimesFM, Chronos, Chronos-2, FlowState, Kairos, TiRex) |
| `mixtureParams` | `MixtureParams` | Full Student-t mixture distribution (Toto only) |

**Quantiles** express uncertainty as percentile ranges. A tight range means the model is confident; a wide range means the signal is hard to predict. For example, a 10th-90th percentile band gives you a plausible low and high around the forecast line.

**Toto's mixture distribution** is more expressive than quantiles. Instead of a few slices of the distribution you get the full probability density. Access it like this:

```swift
// Toto only - full uncertainty distribution
if let params = forecast.mixtureParams {
    print("Weights:", params.weights)          // mixture component weights
    print("Loc:", params.loc)                  // mean of each component
    print("Scale:", params.scale)              // spread of each component
    print("DF:", params.df)                    // degrees of freedom (> 2)
    print("Forecast mean:", params.mean())     // weighted mean across components
}
```

## Supported Architectures

Any model fine-tuned on one of these architectures can be converted and run — not just the reference checkpoints listed below.

| Architecture | Origin | Design | Output | Max Context | Reference Checkpoint |
|-------------|--------|--------|--------|-------------|----------------------|
| **[Toto](https://huggingface.co/Datadog/Toto-Open-Base-1.0)** | Datadog | Patch transformer + space-wise attention | Student-t mixture | 4 096 | `Datadog/Toto-Open-Base-1.0` |
| **[TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch)** | Google | Decoder-only transformer | 9 quantiles | 16 384 | `google/timesfm-2.5-200m-pytorch` |
| **[Chronos](https://huggingface.co/amazon/chronos-t5-base)** | Amazon | T5 encoder-decoder, tokenized | Sampled quantiles | 512 | `amazon/chronos-t5-base` |
| **[Chronos-2](https://huggingface.co/autogluon/chronos-2-synth)** | AutoGluon | T5 encoder-only + RoPE | 13 quantiles | 8 192 | `autogluon/chronos-2-synth` |
| **[Lag-Llama](https://huggingface.co/time-series-foundation-models/Lag-Llama)** | Rasul et al. | Llama-style, lag features | Student-t distribution | 32 | `time-series-foundation-models/Lag-Llama` |
| **[FlowState](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1)** | IBM | S5 SSM + Legendre decoder | 9 quantiles | 2 048 | `ibm-granite/granite-timeseries-flowstate-r1` |
| **[Kairos](https://huggingface.co/mldi-lab/Kairos_50m)** | MLDI-Lab | T5 encoder-decoder, MoS dynamic patching | 9 quantiles | 2 048 | `mldi-lab/Kairos_50m` |
| **[TiRex](https://huggingface.co/NX-AI/TiRex)** | NX-AI | sLSTM with exponential gating | 13 quantiles | 2 048 | `NX-AI/TiRex` |

## Converting Models

The conversion script downloads a HuggingFace checkpoint, remaps weights to the MLX layout, and saves `config.json` + `.safetensors` files. The CLI flags follow the [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) convention.

```bash
# Install Python dependencies
pip install -r Scripts/requirements.txt

# Convert a model
python Scripts/convert_ts_model.py --hf-path Datadog/Toto-Open-Base-1.0 --mlx-path converted/toto

# Convert and quantize to 4-bit in one step
python Scripts/convert_ts_model.py \
  --hf-path Datadog/Toto-Open-Base-1.0 \
  --mlx-path converted/toto-4bit \
  -q --q-bits 4
```

The model type is auto-detected from the HuggingFace ID. You can also pass `--q-bits 2` or `--q-bits 8`, and `--q-group-size` to control the quantization group size (default: 64).

### Upload to HuggingFace

Add `--upload-repo` to push the converted model to HuggingFace Hub with an auto-generated model card:

```bash
python Scripts/convert_ts_model.py \
  --hf-path Datadog/Toto-Open-Base-1.0 \
  -q --q-bits 4 \
  --upload-repo mlx-community/Toto-Open-Base-1.0-4bit
```


## ModelArena

A cross-platform demo app (macOS + iOS) for comparing models side by side. Load any converted model and run live forecasts on synthetic or real data with rolling MASE scores.

```bash
open Applications/ModelArena/ModelArena.xcodeproj
```

Load models by entering a HuggingFace Hub ID directly in the app, or use the file picker to load a locally converted model.

## Project Structure

```
MLX-Swift-TS/
├── Package.swift                        # Swift package manifest
├── Libraries/
│   └── MLXTimeSeries/
│       ├── Core/                        # TimeSeriesModel protocol, factory, config
│       ├── Models/                      # Chronos, Chronos2, TimesFM, LagLlama,
│       │                                  FlowState, Kairos, TiRex
│       ├── Model/                       # Toto + shared transformer components
│       ├── Layers/                      # Distribution heads
│       ├── Distribution/                # Student-t mixture
│       ├── Preprocessing/               # TimeSeriesInput, CausalPatchScaler
│       └── Inference/                   # TimeSeriesForecaster, KV cache
├── Scripts/
│   ├── convert_ts_model.py              # Model conversion script
│   └── requirements.txt
├── Applications/
│   ├── ModelArena/                      # macOS + iOS model comparison app
│   ├── TotoMonitor/                     # System metrics monitoring demo
│   └── WeatherForecast/                 # Weather forecasting demo
└── Tests/
    └── MLXTimeSeriesTests/
```

## Requirements

- **macOS** 14+ or **iOS** 17+
- **Xcode** 15+
- **Apple Silicon** (M1 or later)
- **Python** 3.10+ (for model conversion only)

## Acknowledgments

This project is built on the shoulders of giants. Huge thanks to:

- **[MLX](https://github.com/ml-explore/mlx)** and **[MLX Swift](https://github.com/ml-explore/mlx-swift)** by Apple - the foundation that makes on-device ML in Swift possible
- **[swift-transformers](https://github.com/huggingface/swift-transformers)** by Hugging Face - tokenization and Hub integration for Swift
- **[mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples)** - patterns and inspiration for building MLX-based Swift apps
- **[mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift)** by [Prince Canuma](https://github.com/Blaizzy) - inspiration for this project's structure and README
- **[Datadog](https://github.com/DataDog)** - [Toto](https://huggingface.co/Datadog/Toto-Open-Base-1.0), the time series foundation model that started it all
- The broader **MLX community** on Hugging Face for making model sharing effortless

## License

MIT License - see [LICENSE](LICENSE) for details.
