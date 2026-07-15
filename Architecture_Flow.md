# System Architecture: Multimodal Multi-Agent Neural Architecture

This document maps out the comprehensive architectural flow and structural breakdown of the **Multimodal Multi-Agent Neural Architecture for Early Patient Deterioration Prediction**. This schema partitions raw heterogeneous clinical inputs into specialized unimodal sub-networks ("agents") before performing gated multimodal fusion weighted by uncertainty quantification.

---

## 🏗️ System Architecture Flow Diagram

```text
                  ┌────────────────────────────────────────────────────────┐
                  │                 RAW DATA INPUT LAYERS                  │
                  └────────────────────────────────────────────────────────┘
                               /               |               \
                              /                |                \
                             v                 v                 v
                 ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
                 │  Triage Vitals,  │ │ Sequential Time- │ │ Sparse/Periodic  │
                 │   Demographics   │ │  Series Vitals   │ │ Laboratory Tests │
                 │ (triage/patients)│ │   (vitalsign)    │ │   (labevents)    │
                 └──────────────────┘ └──────────────────┘ └──────────────────┘
                          │                    │                    │
                          v                    v                    v
                 ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
                 │  Tabular Matrix  │ │ 3D Tensor Window │ │ 3D Tensor Window │
                 │    [N x D_s]     │ │  [N x T x D_v]   │ │  [N x T x D_l]   │
                 └──────────────────┘ └──────────────────┘ └──────────────────┘
                          │                    │                    │
==========================│====================│====================│==========================
UNIMODAL AGENT CORES      v                    v                    v
                     ┌──────────┐         ┌──────────┐         ┌──────────┐
                     │  STATIC  │         │  VITALS  │         │   LABS   │
                     │  AGENT   │         │  AGENT   │         │  AGENT   │
                     │  (MLP)   │         │  (LSTM)  │         │  (LSTM)  │
                     └──────────┘         └──────────┘         └──────────┘
                          │                    │                    │
                          │-- Hidden State     │-- Hidden State     │-- Hidden State
                          │   Embedding        │   Embedding        │   Embedding
                          v                    v                    v
                     ┌──────────┐         ┌──────────┐         ┌──────────┐
                     │ Dropout  │         │    MC    │         │    MC    │
                     │  Layer   │         │ Dropout  │         │ Dropout  │
                     └──────────┘         └──────────┘         └──────────┘
                          │                    │                    │
                          │ Vector             │ Vector             │ Vector
                          │ (Deterministic)    │ + Uncertainty (σ²) │ + Uncertainty (σ²)
                          \                    │                    /
                           \                   v                   /
============================\───────► ┌──────────────────┐ ◄──────/===========================
CENTRAL COORDINATION                 │ CENTRAL NEURAL   │ 
                                      │   COORDINATOR    │ (Gated Multimodal Fusion)
                                      └──────────────────┘
                                               │
                                               v
                                      ┌──────────────────┐
                                      │   Output Layer   │ (Sigmoid Activation)
                                      └──────────────────┘
                                               │
                                               v
                                      ┌──────────────────┐
                                      │ PREDICTED RISK   │ P(Deterioration | Patient State)
                                      └──────────────────┘
```

---

## 🔬 Component Breakdown & Clinical Concepts

### 1. Raw Data Input & Preprocessing Layers
Clinical data points exhibit vast imbalances in temporal frequency, density, and formatting. Rather than crushing these signals into a singular unaligned matrix, the system processes them as three distinct streams:
* **Static Stream ($D_s$):** Extracts non-varying variables from `triage` and `patients`. These establish baseline vulnerabilities (e.g., physiological thresholds at admission and fixed demographics).
* **High-Frequency Sequential Stream ($D_v$):** Extracts bedside vitals from `vitalsign` into regular 1-hour uniform bins over an 8-hour lookback window ($T=8$). Gaps are filled using Last Observation Carried Forward (LOCF) imputation.
* **Sparse Sequential Stream ($D_l$):** Extracts cellular/biochemical markers from `labevents`. This tracks internal metabolic reality (e.g., organ stress, tissue hypoxia) over the same 8-hour horizon ($T=8$).

### 2. Unimodal Agent Cores (Expert Networks)
Each stream is assigned a specialized neural architecture optimized for its dimensional geometry:
* **Static Agent (Multi-Layer Perceptron):** Computes flat feature representations to establish stationary clinical risk profiles.
* **Vitals Agent (Sequential LSTM):** Models high-frequency autonomic patterns over time, extracting representations of acute physiological deterioration.
* **Labs Agent (Sequential LSTM):** Captures multi-hour trends in metabolic biomarkers independently to manage missingness and sparse inputs.

### 3. Uncertainty Quantification via Monte Carlo (MC) Dropout
To ensure safe clinical deployment, the sequential agents do not output fixed, deterministic vectors. 
* By retaining active **Monte Carlo Dropout** layers during both training and operational inference, the temporal sub-networks run multiple forward passes to output stochastic representations containing a feature mean ($\mu$) and an explicit mathematical variance ($\sigma^2$).
* This variance ($\sigma^2$) quantifies the system's absolute *epistemic uncertainty* regarding that specific clinical pathway at that exact hour.

### 4. Central Neural Coordinator (Gated Multimodal Fusion)
The Central Coordinator acts as a master clinical decision gate. 
* It ingests the distinct representations simultaneously. Instead of calculating a naive concatenation, it implements a **gated attention mechanism** mapped to the uncertainty bounds.
* If an agent signals high mathematical uncertainty (e.g., the Labs Agent due to highly sparse or missing lab panel arrays), the coordinator dynamically scales down that agent's attention weight and shifts reliance to the deterministic static bounds and high-frequency vitals stream.
* The final output is put through a standard **Sigmoid Activation Layer** to output a calibrated probability of acute patient deterioration $P(	ext{Deterioration} \mid 	ext{Patient State})$.
