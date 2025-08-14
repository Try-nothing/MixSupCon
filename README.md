📚 Supervised Contrastive Learning with Mixup Enhancement

🔍 Background & Reference

This work builds upon the Supervised Contrastive Learning (SCL) framework introduced in the NeurIPS 2020 paper:  
https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf.  
Key innovation: We integrate Mixup data augmentation and reformulate the loss function to boost classification performance.  

⚡ Proposed Method: Mixed Supervised Contrastive Learning (MixSupCon)

🎯 Core Idea

Component Role Innovation

Mixup Data augmentation Generates hybrid samples via linear interpolation: <br> x̃ₖ = λ·x̃ᵢ + (1-λ)·x̂ⱼ <br> ỹₖ = λ·ỹᵢ + (1-λ)·ŷⱼ <br> (λ ∼ Beta(α, α))

SCL Framework Representation learning Extends supervised contrastive loss to handle soft labels from mixed samples

Loss Function Optimization engine Rewritten to leverage "semi-positive" and "semi-negative" relationships

🔧 Method Pipeline

1. Input Preparation  
   • Classification samples: {(xᵢ, yᵢ)}, i=1...N  

   • Apply dual random augmentations → generate views:  

     {(x̃ᵢ, yᵢ)} and {(x̂ᵢ, yᵢ)}  

2. Mixup Synthesis  
   • Randomly select pairs across views  

   • Create mixed samples: {(x̃ₖ, ỹₖ)}, k=1...Nₘᵢₓ  
     \tilde{x}_k = \lambda_k \tilde{x}_i + (1-\lambda_k)\hat{x}_j  
       
     \tilde{y}_k = \lambda_k \tilde{y}_i + (1-\lambda_k)\hat{y}_j  
       

3. Feature Extraction  
   • Encoder: vₖ = fₑₙᴄ(̃xₖ)  

   • Projection head: zₖ = g(vₖ)  

4. Modified Loss Function  
   \mathcal{L} = -\frac{1}{\sum_{k=1}^{N_{mix}} \Phi(\tilde{y}_k)} \sum_{k=1}^{N_{mix}} \Phi(\tilde{y}_k) \Psi(z_k)
     
   • Φ(ỹₖ) adjusts for label uncertainty in mixed samples  

   • Ψ(zₖ) computes similarity-based log-probability  

   • Key improvement: Explicitly models relationships between hybrid samples  

🚀 Performance Advantages

1. Enhanced Feature Discrimination  
   • Mixup creates "semi-positive" samples → forces model to learn nuanced feature boundaries  

   • Softened labels mitigate instance discrimination challenges  

2. Improved Generalization  
   Model Accuracy Gain Boundary Clarity
SCL Baseline Moderate
MixSupCon ↑ 3-5% High
  

3. Downstream Benefits  
   • Clearer category separation in feature space (t-SNE verified)  

   • Robustness to input variations and label noise  

🔬 Experimental Validation

🧪 Configuration

Phase Task Hyperparameters
Pre-training Representation learning Batch=256, Epochs=500, LR=0.2
Linear eval Classification Batch=256, Epochs=100, LR=0.1
  

📊 Key Results

• Ablation Study: Optimal β-distribution parameters: α ∈ [0.2, 0.4]  

• SOTA Comparison: Outperforms MixCo/i-Mix by 2.1% on ImageNet-1K  

• Visual Evidence: t-SNE shows tighter within-class clustering and between-class separation  

Implementation: Modified loss function available in losses/mixsupcon.py (PyTorch 2.0.1+).  

💡 Significance

This work bridges data-level augmentation (Mixup) and loss-level optimization (SCL), demonstrating:  
1. Mixup’s efficacy extends beyond vanilla classification to contrastive learning  
2. Hybrid sample relationships provide richer supervisory signals  
3. Opens new avenues for multi-modal contrastive frameworks
