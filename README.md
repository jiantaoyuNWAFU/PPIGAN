# PPIGAN
PPIGAN: Generating biologically plausible hard negative samples for protein-protein interaction prediction using GAN

## 📌 Overview
PPIGAN is a GAN-based framework for protein-protein interaction (PPI) prediction.  
It generates biologically plausible hard negative samples to improve model generalization.

## 🧠 Motivation
Constructing reliable negative samples is a key challenge in PPI prediction.  
Traditional random sampling leads to unstable performance.

PPIGAN addresses this by:
- Generating negative samples via Conditional GAN
- Improving discrimination ability of PPI models
- Enhancing generalization performance

## 🏗️ Method

The framework consists of:
- Generator: generates fake protein sequences
- Discriminator: DeepTrio-based PPI predictor

Training process:
- Generator produces negative samples
- Discriminator learns from positive + generated negatives
- Adversarial training improves robustness

## 📊 Results

| Dataset | Accuracy |
|--------|--------|
| Yeast  | 94.68% |
| Human  | 98.22% |

PPIGAN achieves competitive or superior performance compared to:
- PIPR
- DeepTrio
- DeepFE

## 📂 Datasets
- Yeast (DIP)
- Human (HPRD)
- BioGRID
- Virus-human

## ⚙️ Installation

```bash
pip install -r requirements.txt
