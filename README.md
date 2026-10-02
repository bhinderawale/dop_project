# PyTorch Implementation of ASATVN

This repository contains a PyTorch implementation of **Adversarial Self-Attentive Time-Variant Neural Networks (ASATVN)**, based on the paper:

> **"Adversarial Self-Attentive Time-Variant Neural Networks for Multi-Step Time Series Forecasting"**  
> Changxia Gao, Ning Zhang, Youru Li, Yan Lin, and Huaiyu Wan

**Original work:** The ASATVN architecture, methodology, and core techniques implemented in this repository are based on the work of the original authors. Please refer to and cite the original paper when using this implementation.

### About ASATVN

ASATVN is a time-variant neural network designed for multi-step time-series forecasting. Its key contributions include:

1. **Time-variant architecture:** Unlike conventional time-invariant RNN and CNN models, ASATVN generates forecasts using a time-variant neural network.
2. **Interleaved outputs:** Interleaved outputs help improve convergence and mitigate vanishing-gradient problems during long-term forecasting.
3. **Truncated Cauchy self-attention:** The proposed self-attention mechanism makes the network more sensitive to local temporal context within the time series.
4. **Self-attentive discriminators:** Two self-attentive discriminators are incorporated to encourage more realistic and continuous long-term forecasts.
5. **Forecasting performance:** The original paper reports that ASATVN outperforms several state-of-the-art deep learning and statistical forecasting models on the evaluated datasets.

## Repository Structure

- **ASATVN.py** — Main ASATVN (Adversarial Self-Attentive Time-Variant Network) implementation.
- **SATVN.py** — Main SATVN (Self-Attentive Time-Variant Network) implementation.
- **Encoder.py** — Encoder consisting of the Truncated Cauchy self-attention layer, feed-forward network, and Add & Norm layers.
- **Truncated_Cauchy_self_attention_block.py** — Implementation of the Truncated Cauchy self-attention block proposed in the paper.
- **Self_attentive_discriminator.py** — Implementation of the self-attentive discriminator.
- **train.py** — Training utilities for ASATVN.
- **evaluate.py** — Evaluation and training utilities.
- **calculateError.py** — Functions for calculating forecasting error metrics.
- **mse_loss.py** — Mean squared error loss implementation.
- **optimizer.py** — Adam optimizer implementation with weight-decay correction.
- **dataHelpers.py** — Dataset generation, preprocessing, and formatting utilities.
- **demo.py** — Example script for training and evaluating ASATVN on the Water Usage dataset.

## Data

The datasets used in the original work can be obtained from the following sources:

- **Water Usage and other time-series datasets:**  
  Rob J. Hyndman's Time Series Data Library:  
  https://robjhyndman.com/tsdl/

- **Additional datasets:**  
  https://arxiv.org/pdf/2012.07436.pdf

The original paper uses these datasets to evaluate ASATVN's ability to model both short-term patterns and long-term trends in sequential data.

## Usage

To train and evaluate ASATVN on the Water Usage dataset, run:

```bash
python demo.py
```

## Requirements

The original implementation was developed using:

- Python 3.6
- PyTorch 1.2.0
- NumPy 1.14.6

Newer versions of these dependencies may require minor compatibility changes.

## Citation

If you use the ASATVN architecture or ideas from this repository, please cite the original paper:

```bibtex
@article{gao2021adversarial,
  title={Adversarial Self-Attentive Time-Variant Neural Networks for Multi-Step Time Series Forecasting},
  author={Gao, Changxia and Zhang, Ning and Li, Youru and Lin, Yan and Wan, Huaiyu},
  journal={...},
  year={2021}
}
```

Please verify the bibliographic details against the original publication before using this citation in academic work.

## Acknowledgement

This repository is based on the work of **Changxia Gao, Ning Zhang, Youru Li, Yan Lin, and Huaiyu Wan**, who proposed the ASATVN architecture and the underlying methods described in:

**"Adversarial Self-Attentive Time-Variant Neural Networks for Multi-Step Time Series Forecasting."**

All credit for the original ASATVN methodology, architecture, and research contributions belongs to the original authors. This repository provides a PyTorch implementation and may contain additional modifications or optimizations made for experimentation and research purposes.
