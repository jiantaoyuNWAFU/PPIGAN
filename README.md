# PPIGAN

**PPIGAN: Prediction of Protein-Protein Interactions Using
Generative Adversarial Networks**

---

## 📌 Overview

Protein–Protein Interaction (PPI) prediction is highly dependent on the quality of negative samples, while random sampling often leads to unstable performance. To address this issue, PPIGAN employs a conditional generative adversarial network (CGAN) to generate hard and realistic negative samples, enabling the model to learn more discriminative interaction features through adversarial training. Experimental results show that PPIGAN achieves 94.68% and 98.22% accuracy on yeast and human datasets, outperforming or matching state-of-the-art methods.

---

## Version

1.0

---

## Authors

Zhang, Xue and Geng etc.

Contact Email: jiantao.yu@nwafu.edu.cn

Repository URL: https://github.com/jiantaoyuNWAFU/PPIGAN

---

## 📂 Datasets

We use multiple benchmark datasets for evaluation:

### Dataset structure

    data/
    ├── Biogrid-human/
    ├── human-pipr/
    ├── virus-human interaction dataset/
    └── yeast core dataset from PIPR/

> ⚠️ Note: Due to the large size of the Biogrid-human dataset, please download it from: [https://pan.nwafu.edu.cn/share/06bbc555338f4b00db222f5c9d](https://pan.nwafu.edu.cn/share/06bbc555338f4b00db222f5c9d)
---

## ⚙️ Installation

```bash
git clone https://github.com/jiantaoyuNWAFU/PPIGAN.git
cd PPIGAN
pip install -r requirements.txt
```

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


