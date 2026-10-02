# Efficient Time-Series LLMs for Edge Deployment

This repository contains implementations and research experiments on **efficient Time-Series Large Language Models (LLMs)**, with a focus on reducing computational cost and memory footprint for deployment on resource-constrained and edge devices.

The work builds upon several existing open-source research projects, particularly **ASATVN**, **Informer**, and **ATOM**, and explores techniques such as **outlier preservation, low-bit quantization, and sparsity** to improve the efficiency of these models.

> **Acknowledgement:** The underlying architectures and research contributions described in ASATVN, Informer, and ATOM belong to their respective authors. This repository builds upon those works and introduces additional optimization and experimentation for efficient time-series forecasting.

---

## Research Motivation

Modern time-series forecasting models can achieve strong forecasting performance but often require substantial computational resources and memory. This can make deployment difficult on resource-constrained devices such as edge devices and low-memory GPUs.

The goal of this project is to investigate how existing time-series architectures can be optimized while maintaining forecasting accuracy.

The primary techniques explored in this work include:

- **Low-bit quantization** — reducing the numerical precision of model weights and activations.
- **Outlier preservation** — identifying and preserving important outlier channels that are particularly sensitive to quantization.
- **Sparsity** — removing or avoiding unnecessary computations and parameters.
- **Memory-efficient inference** — reducing the memory requirements of model parameters and intermediate activations.
- **Efficient attention mechanisms** — leveraging architectures designed to reduce the computational complexity of long-sequence forecasting.

The broader objective is to make advanced time-series models more practical for **edge and resource-constrained deployment**.

---

# Research Foundations

This project builds upon the following research works.

## 1. ASATVN

### Adversarial Self-Attentive Time-Variant Neural Networks for Multi-Step Time Series Forecasting

**Authors:** Changxia Gao, Ning Zhang, Youru Li, Yan Lin, and Huaiyu Wan.

ASATVN is a **time-variant neural network architecture** designed for multi-step time-series forecasting.

Unlike conventional time-invariant RNN and CNN architectures, ASATVN explicitly models temporal variation in the forecasting process.

Its major contributions include:

1. **Time-variant architecture**

   ASATVN introduces a time-variant neural network for multi-step forecasting. The original paper discusses the limitations of conventional RNN and CNN models in representing time-varying forecasting behavior.

2. **Interleaved outputs**

   Interleaved outputs are used to improve convergence and help mitigate vanishing-gradient problems during long-term forecasting.

3. **Truncated Cauchy self-attention**

   ASATVN introduces a Truncated Cauchy self-attention mechanism designed to make the network more sensitive to local temporal context.

4. **Self-attentive discriminators**

   Two self-attentive discriminators are incorporated into the adversarial architecture to encourage more realistic and continuous long-term forecasts.

5. **Multi-step forecasting**

   The architecture is designed specifically for forecasting multiple future time steps rather than only predicting the next observation.

### Implementation

This repository contains a PyTorch implementation of the ASATVN architecture and its associated components.

Relevant files include:

- `ASATVN.py`
- `SATVN.py`
- `Encoder.py`
- `Truncated_Cauchy_self_attention_block.py`
- `Self_attentive_discriminator.py`

---

# 2. Informer

### Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

**Authors:** Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.

Informer is a Transformer-based architecture designed specifically for **long-sequence time-series forecasting**.

Standard Transformer self-attention has quadratic complexity with respect to sequence length, which makes it expensive for long time-series inputs. Informer addresses this limitation through several architectural improvements.

### Key Contributions

#### ProbSparse Self-Attention

Informer introduces **ProbSparse attention**, which focuses computation on the most important queries rather than computing full attention across every query-key pair.

This reduces the computational complexity of self-attention from approximately:

\[
O(L^2)
\]

to:

\[
O(L\log L)
\]

where \(L\) is the sequence length.

#### Self-Attention Distilling

Informer progressively reduces the representation size between encoder layers through an attention-distilling mechanism, reducing memory and computational requirements.

#### Generative Style Decoder

Instead of generating predictions one step at a time, Informer uses a generative-style decoder to produce the entire forecasting sequence more efficiently.

### Relevance to This Project

Informer is particularly relevant to this research because it demonstrates how Transformer-based forecasting models can be redesigned to reduce the computational cost of **long-sequence time-series forecasting**.

The Informer architecture and ideas serve as one of the foundations for the efficiency experiments conducted in this project.

---

# 3. ATOM

### Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving

**Authors:** Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci.

ATOM is a framework for **low-bit quantization of Large Language Models** aimed at improving inference efficiency while maintaining model accuracy.

The work addresses the significant memory and computational requirements of LLM inference through techniques such as:

- Low-bit weight quantization
- Low-bit activation quantization
- Fine-grained group quantization
- Mixed-precision quantization
- Dynamic activation quantization
- KV-cache quantization
- Efficient GPU kernels

A key idea behind ATOM is that aggressive quantization does not necessarily need to be applied uniformly. Different parts of a model can have different sensitivities to quantization, allowing precision to be allocated selectively.

### Relevance to This Project

ATOM provides important concepts for the quantization component of this research, particularly around:

- Group-wise quantization
- Weight and activation quantization
- Outlier-sensitive quantization
- Memory-efficient inference
- Hardware-efficient low-bit computation

These ideas are adapted and explored in the context of **time-series forecasting models**, where the objective is to reduce memory and computation while preserving forecasting performance.

---

# Repository Structure

```text
.
├── ASATVN.py
├── SATVN.py
├── Encoder.py
├── Truncated_Cauchy_self_attention_block.py
├── Self_attentive_discriminator.py
├── dataHelpers.py
├── dataset.py
├── train.py
├── evaluate.py
├── calculateError.py
├── mse_loss.py
├── optimizer.py
├── quant.py
├── probsparse.py
└── demo.py
```

### Main Files

| File | Description |
|---|---|
| `ASATVN.py` | Main ASATVN implementation |
| `SATVN.py` | Self-Attentive Time-Variant Network |
| `Encoder.py` | Encoder containing attention, feed-forward and normalization components |
| `Truncated_Cauchy_self_attention_block.py` | Truncated Cauchy self-attention implementation |
| `Self_attentive_discriminator.py` | Self-attentive discriminator |
| `probsparse.py` | ProbSparse attention implementation |
| `quant.py` | Quantization and quantization utilities |
| `dataset.py` | Dataset processing and loading |
| `dataHelpers.py` | Dataset preparation and formatting |
| `train.py` | Training utilities |
| `evaluate.py` | Evaluation utilities |
| `calculateError.py` | Forecasting error metrics |
| `mse_loss.py` | Mean squared error implementation |
| `optimizer.py` | Adam optimizer implementation |
| `demo.py` | Example training and evaluation script |

---

# Data

The original ASATVN work used several time-series datasets for evaluating short- and long-term forecasting performance.

### Water Usage Dataset

The Water Usage dataset used in the original demonstration can be obtained from:

https://robjhyndman.com/tsdl/

### Additional Datasets

Additional datasets used for evaluating long-term forecasting can be found through:

https://arxiv.org/pdf/2012.07436.pdf

The repository also contains preprocessing utilities for adapting datasets to the input format required by the forecasting models.

---

# Optimization Approach

The main research contribution of this project is the exploration of **model optimization techniques for resource-constrained deployment**.

The optimization pipeline focuses on three major components.

## 1. Outlier Preservation

Quantization can introduce significant errors for outlier values or channels.

Instead of treating every parameter identically, important outlier channels are identified and preserved with higher precision where necessary.

This provides a compromise between:

- aggressive compression
- numerical accuracy
- forecasting performance

---

## 2. Quantization

Model weights and activations are converted from higher-precision representations to lower-bit representations.

The project explores techniques including:

- Per-channel quantization
- Group-wise quantization
- Per-group quantization
- Weight quantization
- Activation quantization
- Outlier-aware quantization

The objective is to reduce:

\[
\text{Memory Footprint}
\]

and

\[
\text{Computational Cost}
\]

while minimizing degradation in forecasting accuracy.

---

## 3. Sparsity

Sparsity is introduced to reduce unnecessary computation by exploiting parameters or operations that contribute relatively little to the final output.

The goal is to reduce the effective computational workload while retaining the important information required for accurate forecasting.

---

# Evaluation

The models can be evaluated using several time-series forecasting metrics, including:

- **MSE** — Mean Squared Error
- **MAE** — Mean Absolute Error
- **NRMSE** — Normalized Root Mean Squared Error
- **SMAPE** — Symmetric Mean Absolute Percentage Error
- **MASE** — Mean Absolute Scaled Error

In addition to forecasting accuracy, the optimization experiments focus on measuring:

- Model memory footprint
- GPU memory consumption
- Computational requirements
- Inference efficiency
- Impact of quantization
- Impact of sparsity
- Accuracy degradation after compression

The goal is not simply to minimize model size, but to study the **trade-off between efficiency and forecasting accuracy**.

---

# Usage

Install the required dependencies and run the demonstration:

```bash
python demo.py
```

For training with custom datasets, modify the dataset configuration and model parameters in the corresponding training scripts.

---

# Requirements

The original ASATVN implementation was developed using:

- Python 3.6
- PyTorch 1.2.0
- NumPy 1.14.6

The optimization experiments may require additional Python packages depending on the configuration being used.

Newer versions of PyTorch and NumPy may require minor compatibility changes to the original implementation.

---

# Acknowledgements

This project would not have been possible without the research and open-source implementations provided by the authors of the underlying models.

We gratefully acknowledge:

### ASATVN

Changxia Gao, Ning Zhang, Youru Li, Yan Lin, and Huaiyu Wan for their work on:

**"Adversarial Self-Attentive Time-Variant Neural Networks for Multi-Step Time Series Forecasting."**

### Informer

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang for their work on:

**"Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting."**

### ATOM

Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci for their work on:

**"Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving."**

We thank the authors for making their research and implementations available to the research community.

The original architectures, algorithms, and research contributions remain attributed to their respective authors. This repository builds upon these works and explores additional optimization techniques for efficient time-series forecasting.

---

# Citation

If you use the original architectures or ideas from these works, please cite the corresponding papers.

### ASATVN

```bibtex
@article{gao2021adversarial,
  title={Adversarial Self-Attentive Time-Variant Neural Networks for Multi-Step Time Series Forecasting},
  author={Gao, Changxia and Zhang, Ning and Li, Youru and Lin, Yan and Wan, Huaiyu}
}
```

### Informer

```bibtex
@inproceedings{zhou2021informer,
  title={Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  author={Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

### ATOM

```bibtex
@inproceedings{zhao2024atom,
  title={Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving},
  author={Zhao, Yilong and Lin, Chien-Yu and Zhu, Kan and Ye, Zihao and Chen, Lequn and Zheng, Size and Ceze, Luis and Krishnamurthy, Arvind and Chen, Tianqi and Kasikci, Baris},
  year={2024}
}
```

> **Note:** Please verify the final BibTeX metadata against the official publications before using these entries in an academic paper.

---

# Disclaimer

This repository is intended for research and educational purposes.

The implementation incorporates and builds upon previously published research. The original authors retain credit for their respective architectures and methodologies. Any additional optimization techniques, modifications, experiments, benchmarks, or extensions introduced in this repository are part of the present research work.
