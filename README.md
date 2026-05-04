# PPIGAN

**PPIGAN: Generating biologically plausible hard negative samples for protein-protein interaction prediction using GAN**

---

## 📌 Overview

PPIGAN is a GAN-based framework designed for protein-protein interaction (PPI) prediction.  
It focuses on generating biologically plausible **hard negative samples**, which significantly improves model generalization and robustness.

Unlike traditional approaches that rely on random negative sampling, PPIGAN leverages a **conditional generative adversarial network (CGAN)** to construct more informative negative samples.

---

## 🧠 Motivation

Constructing reliable negative samples is a fundamental challenge in PPI prediction:

- ❌ Random negative sampling → introduces noise
- ❌ Easy negatives → weak supervision signal
- ❌ Poor generalization across datasets

PPIGAN addresses these issues by:

- ✅ Generating **hard negatives** via Conditional GAN
- ✅ Aligning generated sequence distribution with real proteins
- ✅ Improving discrimination ability of PPI models

---

## 🏗️ Framework

PPIGAN consists of two main components:

### 🔹 Generator
- Takes a protein sequence and random noise as input
- Generates fake protein representations
- Learns to mimic real protein distribution

### 🔹 Discriminator (DeepTrio-based)
- Predicts whether a protein pair interacts
- Distinguishes real vs generated samples

### 🔄 Training Strategy

1. Train discriminator with real positive samples
2. Generator produces fake protein sequences
3. Discriminator learns to classify:
   - real positive pairs
   - generated negative pairs
4. Adversarial training improves both components

---

## 📊 Results

| Dataset | Accuracy |
|--------|--------|
| Yeast  | 94.68% |
| Human  | 98.22% |

PPIGAN achieves competitive or superior performance compared to:

- PIPR
- DeepTrio
- DeepFE

---

## 📂 Datasets

We use multiple benchmark datasets for evaluation:

### Dataset structure

    data/
    ├── Biogrid-human/
    ├── human-pipr/
    ├── virus-human interaction dataset/
    └── yeast core dataset from PIPR/

> ⚠️ Note: Due to size and licensing restrictions, raw datasets are not included in this repository.

---

## 📁 Data Format

### Interaction file (TSV)

protein_id_1    protein_id_2    label

### Sequence file (TSV)

protein_id    sequence

### Label definition

- 1 → interacting pair  
- 0 → non-interacting pair  

---

## ⚙️ Installation

```bash
git clone https://github.com/jiantaoyuNWAFU/PPIGAN.git
cd PPIGAN
pip install -r requirements.txt
