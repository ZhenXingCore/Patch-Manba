# 🚢 Patch-Mamba: Ship Trajectory Prediction for Irregular Multivariate Time Series

**Patch-Mamba** is a deep learning framework designed for **real-world ship trajectory prediction** under **irregular sampling, missing observations, and heterogeneous data sources**.

Unlike traditional methods that rely on **spatiotemporal interpolation preprocessing**, Patch-Mamba directly models **irregular multivariate time series**, enabling more realistic and robust trajectory forecasting.

---

## 🔥 Key Features

- **Real-world applicability**  
  Designed for real maritime data from **AIS, BDS, and Radar**, without requiring trajectory reconstruction.

- **Patch-based trajectory modeling**  
  Trajectories are segmented into **multi-scale temporal patches**, capturing local motion patterns and long-range dependencies.

- **Patch Completion Mechanism**  
  A novel **mask-denoising strategy** reconstructs missing trajectory patches, improving robustness to data sparsity.

- **Mamba-based sequence modeling**  
  Integrates **Selective State Space Model (SSM)** for efficient long-sequence modeling with linear complexity.

- **Strong performance**  
  Achieves competitive results on multiple real-world datasets, especially under **irregular and incomplete trajectories**.

---

## 🧠 Method Overview

Patch-Mamba consists of three core components:

1. **Patch Embedding Module**
   - Sliding time window segmentation
   - Continuous time embedding
   - Graph Attention for intra-patch modeling

2. **Patch Completion Module**
   - Masked patch reconstruction
   - Handles missing observations and empty patches

3. **Mamba Backbone**
   - Models global trajectory dependencies
   - Efficient long-sequence learning via SSM

This design enables the model to jointly learn:

- local spatiotemporal dynamics
- global trajectory evolution
- missing data reconstruction

---

## 📊 Supported Data

The model is designed for **multi-source maritime trajectory data**, including:

- AIS (Automatic Identification System)
- BDS (Beidou Navigation System)
- Radar

It is especially suitable for:

- Irregular time intervals
- Missing trajectory points
- Multi-sensor heterogeneous data

---

## 📦 Dataset

We release the **MRST dataset**, a real-world ship trajectory dataset:

- Sources: AIS, BDS, Radar
- Includes irregular and complete trajectories
- Covers 2h / 4h / 6h trajectory sequences

---

## 🚀 Why Patch-Mamba?

Traditional trajectory prediction methods usually assume:

- regular sampling
- fully observed sequences

But in real-world maritime scenarios:

- data is irregular
- observations are missing
- interpolation may introduce noise

Patch-Mamba directly addresses these challenges by:

- eliminating preprocessing assumptions
- modeling raw trajectory dynamics
- improving prediction accuracy in real scenarios

---

## 📌 Applications

- Maritime traffic prediction
- Collision avoidance
- Route planning
- Intelligent ocean monitoring

---

## 📎 Paper

If you find this project useful, please consider citing our work.

---

## ⭐ TODO

- [ ] Release training scripts
- [ ] Add pretrained models
- [ ] Support more trajectory datasets
- [ ] Extend to multi-agent prediction
