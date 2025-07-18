# 🚗 Trajectory Smoothing Strategy Optimization System

## 📋 Overview

This project provides a complete trajectory smoothing strategy optimization system designed for handling noise in vehicle trajectory data. Traditional trajectory smoothing methods often assume that smoothing does not affect the accuracy of Surrogate Safety Measures (SSMs), but this assumption overlooks the fact that inappropriate smoothing may distort safety-critical kinematic features, thereby affecting evaluation outcomes.

 Inspired by elastoplasticity theory, we propose an adaptive smoothing method to dynamically identify the optimal smoothing intensity, which suppresses the propagation of noise and minimizes its distortion on SSMs, achieving a balance where the signal bears just enough regularization to suppress noise without compromising safety-critical kinematic features.

## ✨ Features

- 💻 **Multi-Data Source Support**: Supports HighD dataset and simulated data
- 📊 **Adaptive Optimization**: Elastoplasticity theory-inspired adaptive smoothing parameter optimization
- 🎯 **Precise Smoothing**: Various smoothing methods including Gaussian filter, ensuring low RMSE (<0.17)
- 📈 **Visualization Analysis**: Complete optimization process and result visualization
- 🔧 **Modular Design**: Clear class structure, easy to extend and maintain

## 🛠️ Installation

### 📋 Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- SciPy

### 🔧 Installation Steps

Install the required packages directly:

```bash
pip install numpy pandas matplotlib scipy
```

## 🚀 Quick Start

### 💻 Basic Usage

```python
from SmoothStrategy_CN import *

# Create processor instances
processor = TrajectoryProcessor()
optimizer = OptimizationEngine()
visualizer = Visualizer()

# Load or create data
trajectory, data_source = load_or_create_data(use_sample=True, vehicle_id=323)

# Process trajectory data
processed_data = processor.process_trajectory(trajectory, noise_std=0.25)

# Parameter optimization
smoothing_func = SmoothingMethods.gaussian_filter_method
param_ranges = [np.arange(2, 20, 0.2)]
deltas = [0.2]
param_names = ['sigma']

results = optimizer.optimize_parameters(
    processed_data, 
    smoothing_func, 
    param_ranges, 
    deltas,
    stability_method='weighted'
)

# Visualize results
fig = visualizer.plot_optimization_results(results, param_names)
plt.show()
```

### 📊 Using HighD Data

```python
# Configure HighD data path
HIGHD_DATA_PATH = 'path/to/highd/data/'

# Load real data
trajectory, data_source = load_or_create_data(
    highd_path=HIGHD_DATA_PATH,
    use_sample=False,
    vehicle_id=323
)
```

## 🏗️ System Architecture

### 🧩 Core Modules

#### 1. 🔄 TrajectoryProcessor
- **Function**: Trajectory data preprocessing, including noise addition, downsampling, etc.
- **Main Methods**:
  - `add_noise()`: Add Gaussian noise
  - `downsample_trajectory()`: Trajectory downsampling
  - `process_trajectory()`: Complete processing pipeline

#### 2. 🎛️ SmoothingMethods
- **Function**: Provide various trajectory smoothing algorithms
- **Supported Methods**:
  - Gaussian Filter
  - Moving Average (extensible)
  - Savitzky-Golay Filter (extensible)
  - Butterworth Filter (extensible)

#### 3. ⚙️ OptimizationEngine
- **Function**: Core algorithm for parameter optimization
- **Main Methods**:
  - `calculate_local_variance()`: Calculate local variance
  - `calculate_stability()`: Calculate stability metric
  - `optimize_parameters()`: Main parameter optimization function

#### 4. 📊 Visualizer
- **Function**: Result visualization and analysis
- **Main Charts**:
  - Trajectory comparison plots
  - Optimization result plots
  - Smoothing effect comparison plots

## 🔬 Algorithm Principles

### 💡 Core Concept

This system is inspired by **elastoplasticity theory** and proposes an adaptive smoothing method. The core idea is to find the optimal smoothing intensity that allows trajectory data to bear "just enough" regularization to suppress noise without compromising safety-critical kinematic features.

### 🔗 Error Chain Analysis

The system constructs a complete error chain to analyze how measurement noise propagates through smoothing and distorts Surrogate Safety Measures (SSMs):

```
Raw Trajectory → Measurement Noise → Smoothing → Kinematic Features → SSMs Calculation → Safety Assessment
```

### 🎯 Optimization Objective Function

The system's core is based on **stability metric** optimization:

```
S(τ) = Σ(w_i × σ_i)
```

Where:
- `S(τ)`: Stability metric
- `w_i`: Weights (based on acceleration mean)
- `σ_i`: Local acceleration variance
- `τ`: Smoothing parameter

### 📈 Gradient Calculation

Gradient calculation formula:
```
∇S(τ) = [S(τ) - S(τ + Δτ)] / S(τ)
```

### ✅ Convergence Criteria

- **Valid Proportion**: Proportion of acceleration and jerk within reasonable range ≥ 99%
- **Gradient Change**: Gradient <0.03, Consecutive gradient change rate < 0.003

### 📊 Performance Validation

Validation results on the HighD dataset show:
- **RMSE for position, velocity, and acceleration** < 0.17
- **Reliability across four representative SSMs**
- **High reliability for acceleration-sensitive SSMs**

## 📁 Data Format

### 📥 Input Data Format

Trajectory data should contain the following columns:

```csv
frame,id,x,y,width,height,xVelocity,yVelocity,xAcceleration,yAcceleration,laneId
```

Note: width, height, xVelocity, yVelocity, xAcceleration, yAcceleration, laneId are optional parameters

### 📝 Sample Data

The system automatically generates sample data containing:
- 200 time steps
- Multi-phase motion (acceleration, constant speed, deceleration)
- Vehicle ID: 323

## ⚙️ Configuration Parameters

### 🎛️ Main Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_frequency` | 25 | Original sampling frequency (Hz) |
| `target_frequency` | 10 | Target sampling frequency (Hz) |
| `time_window` | 10 | Time window size |
| `noise_std` | 0.25 | Noise standard deviation |
| `convergence_threshold` | 0.003 | Convergence threshold |

### 🔧 Optimization Parameters

```python
# Gaussian filter parameter range
param_ranges = [np.arange(2, 20, 0.2)]  # sigma range: 2-20, step 0.2
deltas = [0.2]  # Gradient calculation step size
```

## 📊 Result Analysis

### 📈 Visualization Output

1. **Trajectory Comparison**: Original vs noisy vs smoothed trajectories
2. **Optimization Process**: Gradient changes and convergence process
3. **RMSE Analysis**: Error analysis under different parameters
4. **Smoothing Effect Comparison**: Multi-parameter effect comparison

### 📏 Performance Metrics

- **RMSE**: Root Mean Square Error
- **Valid Proportion**: Proportion of reasonable values
- **Stability Metric**: System stability measure

## 🔧 Extensions

### ➕ Adding New Smoothing Methods

```python
class SmoothingMethods:
    @staticmethod
    def your_custom_filter(data, param1, param2):
        """Custom filtering method"""
        # Implement your filtering algorithm
        return filtered_data
```

### 🎯 Custom Optimization Strategy

```python
class OptimizationEngine:
    def your_custom_optimization(self, data, smoothing_func, params):
        """Custom optimization strategy"""
        # Implement your optimization algorithm
        return optimized_params
```

## 📂 File Structure

```
trajectory-smoothing-optimization/
├── SmoothStrategy_CN.ipynb          # Main program (Jupyter version)
├── SmoothStrategy_EN.ipynb          # Main program (Jupyter version)
├── sample_trajectory.csv            # Sample trajectory data
├── trajectory_id_323.csv            # Processed trajectory data
└──README.md                        # Documentation
```

## 📊 Example Results

### 📈 Before and After Optimization (Only calculable when ground truth is available)

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Velocity RMSE | 2.45 | 0.87 | 64.5% |
| Acceleration RMSE | 4.12 | 1.23 | 70.1% |
| Valid Proportion | 76.3% | 99.2% | 22.9% |

### 🎯 Optimal Parameter Example

```
Data source: highd
Optimal parameters: [7.4]
```

## ❓ FAQ

### Q1: How to handle large-scale data?

A: The system supports batch processing, you can optimize memory usage by adjusting the `time_window` parameter.

### Q2: What data formats are supported?

A: Currently supports CSV format, other formats need to be preprocessed to standard format.

### Q3: How to choose appropriate smoothing parameter range?

A: It's recommended to start with a smaller range (e.g., 2-10) and adjust based on data characteristics.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{trajectory_smoothing_optimization,
  title={Trajectory Smoothing Strategy Optimization Method},
  author={Chengmin Li},
  year={2024},
  url={https://github.com/tuqing123/Smoothstrategy}
}
```

## 📞 Contact

- Author: [Chengmin Li]
- Email: [lchengmin9@google.com]
- Project Link: [tuqing123/Smoothstrategy](https://github.com/tuqing123/Smoothstrategy)

## 📝 Changelog

### v1.0.0 (2025-07-17)
- Initial release
- Implemented basic trajectory smoothing functionality
- Support for Gaussian filter optimization
- Complete visualization system

### v1.1.0 (Planned)
- Add more smoothing methods
- Support for parallel processing
- Performance optimization

---

## 🙏 Acknowledgments

Thanks to all contributors and the following projects:
- HighD dataset providers
- SciPy development team
- Matplotlib community