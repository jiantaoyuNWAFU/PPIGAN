# PPIGAN

**PPIGAN: Prediction of Protein-Protein Interactions Using
Generative Adversarial Networks**

---

## 📌 Overview

Protein–Protein Interaction (PPI) prediction is highly dependent on the quality of negative samples, while random sampling often leads to unstable performance. To address this issue, PPIGAN employs a conditional generative adversarial network (CGAN) to generate hard and realistic negative samples, enabling the model to learn more discriminative interaction features through adversarial training. Experimental results show that PPIGAN achieves 94.68% and 98.22% accuracy on yeast and human datasets, outperforming or matching state-of-the-art methods.

---

## 📦 Version

1.0

---


## 📂 Datasets

We use multiple datasets for evaluation:

### Dataset structure

    data/
    ├── Negatome1
    ├── Negatome2
    ├── Biogrid-human
    ├── human-pipr
    ├── virus-human
    └── yeast

> ⚠️ Note: Due to the large size of the Biogrid-human dataset, please download it from: [https://pan.nwafu.edu.cn/share/06bbc555338f4b00db222f5c9d](https://pan.nwafu.edu.cn/share/06bbc555338f4b00db222f5c9d)
---

## ⚙️ Installation

```bash
git clone https://github.com/jiantaoyuNWAFU/PPIGAN.git
cd PPIGAN

conda create -n ppigan python=3.9 -y
conda activate ppigan

pip install torch torchvision torchaudio
pip install numpy scikit-learn matplotlib tqdm
```

## 🚀 Usage

### 1. Train PPIGAN

```
bash scripts/run_yeast.sh 
```

### 2. Train PPIGAN_5fold

```
bash scripts/run_yeast_5fold.sh 
```

---

## 📊 Output

After training, results are saved under:

| Directory      | Description |
|---------------|------------|
| checkpoints  | Saved model weights (e.g., D_best_acc.pth, G_best_acc.pth) |
| datasets     | Cached train/test splits (.pth files) |
| fake_samples | Generated negative samples (tensor format) |
| fasta_list   | Generated protein sequences (FASTA format) |
| logs         | Training logs |

---

## ✒️ Citation

If you use PPIGAN in support of your work, please cite:

Zhang X.\*, Xue S.\*, Geng J.\*, Wen X., Lai L., Huang L., and Yu J.\# 2026. PPIGAN: Prediction of Protein-Protein Interactions Using Generative Adversarial Networks. Journal of Computational Biology (Accepted).

Contact Email: jiantao.yu@nwafu.edu.cn  
Repository URL: https://github.com/jiantaoyuNWAFU/PPIGAN



