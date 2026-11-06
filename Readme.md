# Physics-Informed Quantum Graph Attention Network (PI-QGAT)

### A Hybrid Quantum-Classical Graph Learning Framework for Particle Transverse Momentum Prediction

## Overview

This project introduces **Physics-Informed Quantum Graph Attention Network (PI-QGAT)**, a hybrid quantum-classical graph learning framework designed to predict particle transverse momentum (\(p_T\)) from detector measurements in the CMS Trigger Dataset.

The model combines Graph Neural Networks (GNNs), physics-informed attention, and a parameterized quantum circuit to learn interactions between detector stations. A 4-qubit quantum circuit is integrated into the graph message-passing mechanism to capture pairwise relationships between detector stations through quantum interactions and entanglement.

Beyond developing the architecture, this project investigates the contribution of its quantum and classical components through extensive ablation experiments and controlled quantum-versus-classical comparisons.

## Key Features

- **Hybrid Quantum-Classical Architecture:** Integrates a parameterized quantum circuit into a graph attention framework.
- **Physics-Informed Attention:** Incorporates detector geometry through angular and bending-angle differences.
- **Quantum Interaction Block:** Uses a 4-qubit variational circuit with trainable rotation gates, data re-uploading, and adaptive IsingZZ entanglement.
- **Graph Message Passing:** Models interactions among four detector stations using directed pairwise connections.
- **Optimized JAX Simulator:** Implements a fast, closed-form quantum circuit simulator and verifies its outputs against PennyLane.
- **Extensive Ablation Studies:** Investigates the contributions of quantum interactions, physics priors, circuit configurations, detector features, and graph message passing.

## Architecture

PI-QGAT processes measurements from four detector stations, represented as nodes in a fully connected directed graph.

Each station contains seven input features:

| Feature | Description |
|---|---|
| Phi | Angular coordinate |
| Theta | Polar coordinate |
| BendingAngle | Track bending measurement |
| TimeInfo | Timing information |
| RingNumber | Detector ring information |
| Front | Detector front-layer indicator |
| Mask | Detector measurement mask |

Each node is projected into a 4-dimensional representation, matching the four qubits used in the quantum interaction block.

### Model Components

**1. Node and Pairwise Projection**

- Projects seven station features into four-dimensional node embeddings.
- Combines receiver and sender embeddings to construct pairwise interaction inputs.
- Generates four rotation angles for each detector-station pair.

**2. Quantum Interaction Block**

The quantum circuit consists of:

- Four qubits.
- Parameterized RY and RZ rotation gates.
- Ring-based IsingZZ entanglement.
- Data re-uploading between variational layers.
- Pauli-Z expectation measurements.

The quantum circuit produces four expectation values for each station pair, which are used in graph attention and message passing.

**3. Physics-Informed Graph Attention**

The attention mechanism combines quantum outputs with a physics-based prior derived from:

- Absolute bending-angle differences.
- Angular differences between detector stations.

This allows the model to consider both learned quantum interactions and detector-related physical information when calculating attention weights.

**4. Graph Message Passing**

The model performs one round of attention-weighted message passing across the four detector stations. Quantum interaction outputs contribute to both attention calculation and message construction.

**5. Prediction Head**

The aggregated graph representation is passed through a classical neural network head (4 → 6 → 1) to predict the normalized log-transformed transverse momentum.

## Quantum Circuit Design

The quantum interaction block follows a two-layer variational circuit:

1. Encode pairwise features using RY rotations.
2. Apply trainable RZ and RY rotations.
3. Perform adaptive ring-based IsingZZ entanglement.
4. Re-upload the input angles using RY rotations.
5. Apply the second variational rotation layer.
6. Measure Pauli-Z expectation values from all four qubits.

The adaptive entanglement angles are generated using a trainable classical network, allowing the entangling interactions to depend on the input detector-pair features.

## Dataset and Preprocessing

**Dataset:** CMS Trigger Dataset

**Source:** [Kaggle – CMS Trigger Dataset](https://www.kaggle.com/datasets/prateekagnihotri/cms-trigger-dataset)

The dataset contains 1,179,356 events.

Preprocessing includes:

- Extraction of detector measurements from four selected stations.
- Reconstruction of transverse momentum using \(p_T = |1/(q/p_T)|\).
- Removal of invalid and non-finite observations.
- Feature standardization using training-set statistics.
- Logarithmic target transformation using \(\log(1+p_T)\).
- Construction of a fully connected directed graph with 12 edges.

**Data split:**

| Subset | Percentage |
|---|---:|
| Training | 20% |
| Validation | 20% |
| Testing | 60% |

All dataset partitions are retained.

## Experimental Design: Ablation Studies

A major part of this project involves systematic ablation experiments to understand the contribution of individual architectural components.

The experiments investigate the following aspects:

### 1. Quantum vs. Classical Comparison

Compared the quantum interaction block against parameter-matched and reduced-parameter classical alternatives to investigate the contribution of quantum interactions.

### 2. Physics Prior Ablation

Compared physics-informed attention with attention without the physics prior to investigate the importance of detector geometry.

### 3. Entanglement Topology

Investigated different quantum entanglement configurations:

- Ring
- Linear/chain
- All-to-all
- No entanglement

### 4. Quantum Circuit Capacity

Examined different numbers of qubits and circuit depths to investigate the effects of circuit dimensionality and expressiveness.

### 5. Entangling Gate Ablation

Compared different entangling mechanisms, including adaptive RZZ, CNOT, adaptive CRX, and fixed-angle RZZ configurations.

### 6. Quantum Circuit Initialization

Investigated different parameter initialization strategies, including uniform, small-uniform, normal, and zero initialization.

### 7. Feature Ablation

Performed leave-one-feature-out retraining and permutation importance analysis to evaluate the contribution of individual detector features.

### 8. Graph Attention Ablation

Investigated the role of quantum outputs in attention calculation and message construction, including uniform attention, quantum-message-only, quantum-attention-only, and no-quantum configurations.

### 9. Message-Passing Ablation

Evaluated different message aggregation and graph configurations, including:

- Sum and max aggregation.
- Residual connections.
- Mean and max graph readout.
- One versus two message-passing rounds.
- Pairwise versus sender-only interactions.
- Removal of message passing.

### 10. Quantum Circuit Distillation

Investigated whether the trained quantum interaction block could be approximated by classical neural networks with different hidden-layer capacities, using block-level knowledge transfer and prediction-level distillation.

## Results

The main PI-QGAT model contains **144 trainable parameters**.

In the recorded test evaluation:

| Metric | Result |
|---|---:|
| Test \(R^2\) (log1p scale) | 0.8395 |
| Test MAE (log1p scale) | 0.2143 |
| Test MSE (log1p scale) | 0.1416 |
| Test RMSE (log1p scale) | 0.3763 |
| Test MAE (\(p_T\)) | 11.8517 GeV |

### Quantum vs. Classical Ablation

The recorded parameter-matched comparison produced the following results:

| Model | Parameters | Test \(R^2\) |
|---|---:|---:|
| PI-QGAT (Quantum) | 144 | 0.8395 |
| Classical-36 | 144 | 0.6150 |
| Classical-16 | 124 | 0.5750 |
| No-Quantum | 72 | 0.3904 |

The parameter-matched experiment investigates whether the quantum interaction block provides a different learning capability from a classical block with the same parameter budget.

*Note: These comparison results are from the recorded seed-42 experiment and should be interpreted as preliminary evidence rather than multi-seed statistical validation.*

## Implementation and Optimization

The project uses an optimized JAX-based implementation of the four-qubit quantum circuit.

Key implementation details:

- Closed-form quantum state simulation.
- JAX JIT compilation and automatic differentiation.
- XLA-based training optimization.
- Fused quantum circuit operations.
- PennyLane-based circuit correctness verification.
- Best-validation-checkpoint model selection.

The optimized simulator was numerically verified against PennyLane's `default.qubit` simulator.

## Technologies Used

- **Python**
- **JAX** – Differentiable quantum simulation and optimization
- **PyTorch** – Data handling and neural network components
- **PennyLane** – Quantum circuit construction and verification
- **NumPy & Pandas** – Numerical computation and data preprocessing
- **Scikit-learn** – Data splitting and evaluation metrics
- **KaggleHub** – Dataset acquisition

## Running the Project

The experiments were developed in Google Colab-compatible Jupyter notebooks.

1. Open the notebook in Google Colab.
2. Run the package installation and import cell.
3. Run the dataset acquisition and preprocessing cell.
4. Execute the PI-QGAT model training cell.
5. Run the individual ablation experiment sections to reproduce the comparisons.

PennyLane and KaggleHub are installed automatically by the setup cell.

## Research Objective

The central objective of this project is to investigate how parameterized quantum circuits and physics-informed graph learning can work together for particle-level prediction tasks.

Rather than assuming that quantum components automatically provide an advantage, the project uses controlled ablation experiments and parameter-matched classical comparisons to investigate their actual contribution to the overall architecture.

---

**Project Focus:** Quantum Machine Learning | Hybrid Quantum-Classical Models | Graph Neural Networks | Physics-Informed Learning | Particle Physics
