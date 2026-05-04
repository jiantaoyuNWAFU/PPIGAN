# PPIGAN

**PPIGAN: Prediction of Protein-Protein Interactions Using
Generative Adversarial Networks**

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

PPIGAN is a conditional GAN framework for PPI prediction, designed to generate **biologically plausible hard negative samples**.

### 🔹 Generator
- Takes a protein sequence and noise as input  
- Generates realistic protein representations  
- Learns the distribution of real proteins  

### 🔹 Discriminator (DeepTrio-based)
- Takes a pair of protein representations  
- Predicts interaction probability  
- Distinguishes real pairs from generated negatives  

### 🔄 Training Strategy

- Train on real positive protein pairs  
- Generate hard negative samples via the generator  
- Perform adversarial training between generator and discriminator  

### ✨ Key Idea

Instead of random sampling, PPIGAN generates **hard and realistic negative samples**, improving model generalization.

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

## ⚙️ Installation

```bash
git clone https://github.com/jiantaoyuNWAFU/PPIGAN.git
cd PPIGAN
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train PPIGAN

```bash
python train_cgan.py \
  --interaction_data "./data/Biogrid-human/third_human_MV_pair.tsv" \
  --sequence_data "./data/Biogrid-human/double_human_MV_database.tsv" \
  --epoch 50 \
  --batch_size 16 \
  --cuda
```

### 2. Five-fold cross-validation

```bash
python train_original_5_fold.py \
  --interaction_data "./data/yeast core dataset from PIPR/protein.actions.tsv" \
  --sequence_data "./data/yeast core dataset from PIPR/protein.dictionary.tsv" \
  --epoch 50 \
  --batch_size 64 \
  --cuda
```

### 3. Independent test

```bash
python independent_test.py \
  --interaction_data "./data/virus-human interaction dataset/test_pair.tsv" \
  --sequence_data "./data/virus-human interaction dataset/test_sequence.tsv" \
  --d_pth "./checkpoints/D_best.pth" \
  --batch_size 32 \
  --cuda
```

### 4. Run on CPU

If GPU is unavailable, remove `--cuda`:

```bash
python train_cgan.py \
  --interaction_data "./data/Biogrid-human/third_human_MV_pair.tsv" \
  --sequence_data "./data/Biogrid-human/double_human_MV_database.tsv" \
  --epoch 50 \
  --batch_size 16
```


