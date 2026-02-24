# MLX-Swift-TS

Run Time series foundation models on Apple Silicon.

![Platform](https://img.shields.io/badge/platform-macOS%2014%2B%20%7C%20iOS%2017%2B-lightgrey)
![Swift](https://img.shields.io/badge/Swift-5.9%2B-orange)

## Overview

MLX-Swift-TS is a Swift SDK for running time series foundation models locally on Mac and iOS using [MLX](https://github.com/ml-explore/mlx-swift). Convert PyTorch checkpoints from HuggingFace to MLX format, then run inference - no server required.

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

```swift
import MLXTimeSeries

// Load a converted model
let forecaster = try TimeSeriesForecaster.loadFromDirectory(modelURL)

// Or download directly from HuggingFace Hub
let forecaster = try await TimeSeriesForecaster.loadFromHub(id: "mlx-community/Toto-Open-Base-1.0-mlx")

// Forecast
let input = TimeSeriesInput.univariate(historicalValues)
let prediction = forecaster.forecast(input: input, predictionLength: 64)

print(prediction.mean)       // point forecast [1, 64]
print(prediction.quantiles)  // quantile forecasts (when available)
```

## Supported Models

| Model | Origin | Architecture | Output | Context | HuggingFace ID |
|-------|--------|-------------|--------|---------|----------------|
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

## Data Format

### Input - `TimeSeriesInput`

All models accept a common input with shape **`[B, V, T]`** (batch, variates, time steps).

**Univariate** means a single variable over time -- for example, hourly CPU usage of one server or daily closing price of one stock. Most time series forecasting starts here.

**Multivariate** means multiple related variables measured over the same time period -- for example, CPU usage *and* memory usage of the same server, or temperature *and* humidity from the same sensor. The model can learn cross-variable patterns to improve forecasts.

```swift
import MLXTimeSeries
import MLX

// Univariate -- one variable, e.g. hourly temperature readings
let input = TimeSeriesInput.univariate([72.1, 73.4, 71.8, 70.2, 69.5, ...])

// Multivariate -- two variables measured together, e.g. temperature + humidity
// Shape: [1 batch, 2 variates, 512 time steps]
let series = MLXArray(/* shape: [1, 2, 512] */)
let input = TimeSeriesInput(
    series: series,
    paddingMask: MLXArray.ones([1, 2, 512]),
    idMask: MLXArray.zeros([1, 2])
)
```

### Output - `TimeSeriesPrediction`

```swift
prediction.mean          // [B, V, predictionLength]     - point forecast
prediction.quantiles     // [B, V, predictionLength, Q]  - quantile forecasts (when available)
prediction.mixtureParams // MixtureParams                - full distribution (Toto)
```

## ModelArena

A cross-platform demo app (macOS + iOS) for comparing models side by side. Load any converted model and run live forecasts on synthetic or real data with rolling MASE scores.

```bash
open Applications/ModelArena/ModelArena.xcodeproj
```

On macOS the app auto-loads models from the `converted/` directory. On iOS, use the document picker or enter a HuggingFace Hub ID.

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
│   ├── convert_ts_model.py              # PyTorch -> MLX conversion (all 8 models)
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
