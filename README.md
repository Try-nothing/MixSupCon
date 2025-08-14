## Supervised Contrastive Learning with Mixup (MixSupCon)

This project builds upon the **Supervised Contrastive Learning (SupCon)** framework  
([NeurIPS 2020 paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf))  
and introduces **Mixup** data augmentation to further boost classification accuracy.

### 🔧 Key Improvements

| Component         | Original SupCon                            | Our Modification (MixSupCon)                     |
|-------------------|--------------------------------------------|--------------------------------------------------|
| **Data Pipeline** | Two random augmentations per image         | **+** Mixup between two augmented views          |
| **Loss Function** | `L_sup_out` (Eq. 2 in the paper)           | **Rewritten** to accept soft (mixed) labels      |
| **Contrastive Pairs** | Same-class positives vs. all negatives | **Semi-positive / semi-negative** pairs via soft labels |

### 🧪 Implementation Highlights

1. **Mixup Generation**  
   - For each mini-batch, generate two augmented views `x̃`, `x̂`.  
   - Sample `λ ~ Beta(α, α)` and mix both images and one-hot labels:  
     ```
     x_mixed = λ * x̃ + (1-λ) * x̂
     y_mixed = λ * ỹ   + (1-λ) * ŷ
     ```

2. **Modified Loss**  
   - Replace the original binary mask for positive/negative selection with **soft similarity scores** computed from mixed labels.  
   - The new contrastive loss handles fractional similarities, enabling the model to learn from **inter-class relationships** instead of hard boundaries.

3. **Training Setup** (example)  
   | Hyper-parameter | Value   |
   |-----------------|---------|
   | Mixup α         | 0.8     |
   | Temperature τ   | 0.1     |
   | Batch size      | 256     |
   | Optimizer       | LARS + RMSProp |

### 📈 Results Snapshot (CIFAR-10 / CIFAR-100)

| Method                 | Top-1 Acc. (CIFAR-10) | Top-1 Acc. (CIFAR-100) |
|------------------------|-----------------------|------------------------|
| SupCon (baseline)      | 96.0 %                | 76.5 %                 |
| **MixSupCon (ours)**   | **96.7 %**            | **77.8 %**             |

> Gains are consistent across ResNet-18/34/50 backbones and robust to label noise.

### 🚀 Getting Started

```bash
git clone https://github.com/your-id/MixSupCon.git
cd MixSupCon
pip install -r requirements.txt
python train.py --config configs/cifar10_mixsupcon.yml
