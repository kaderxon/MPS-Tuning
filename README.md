# 🌟 MPS-Tuning
🎉 **Preserve and Sculpt: Manifold-Aligned Fine-Tuning of Vision-Language Models for Few-Shot Learning**

Official PyTorch implementation of "Preserve and Sculpt: Manifold-Aligned Fine-Tuning of Vision-Language Models for Few-Shot Learning" accepted at **ICLR 2026**! 🚀🍾

## 📢 Updates
* **[2026-03-12]** 🏗️ **Code is here!** We have released the dataset configuration files 📊 and the model trainers 🧠! Thank you so much for your patience! ❤️✨

## 📂 About the Trainers
In our `trainers` directory, you will find two versions of the implementation ✌️:
1. 🔬 **The Original Version (`MPSTuning`)**: This is the exact, battle-tested version developed and used by us during our research. 
2. 🤖 **The Claude-Refined Version (`MPSTuning_ClaudeRefined`)**: We used the Claude model to clean up and organize the code for better readability ✨. *A gentle heads-up 💡:* Due to tight schedules ⏰, we haven't comprehensively benchmarked this specific refined version for actual performance. However, we have manually reviewed the code 👀, and it structurally and logically aligns with the methodology described in our paper 📄. 

💬 If you encounter any unexpected behaviors, bugs 🐛, or reproduction issues with either version, please **feel free to contact us**! We are more than happy to help you out. 🤝

## 🛠️ Installation & Usage
Our method is proudly built upon the awesome [CoOp](https://github.com/KaiyangZhou/CoOp) codebase 🧩 and the [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch) library 📚. 

### 📦 Prerequisites
* 🆕 **If you haven't used CoOp/Dassl before:** Please follow the official [CoOp installation instructions](https://github.com/KaiyangZhou/CoOp) to install `Dassl`, set up your environment 🌍, and prepare the datasets first 🖼️.
* ✅ **If you already have Dassl installed and use CoOp:** You are basically ready to go! 🎉

### 🚀 How to Run MPS-Tuning
Integrating our method into the CoOp framework is completely plug-and-play 🔌:
1. 📁 **Move** the trainer files from our `trainers` folder into your `CoOp/trainers/` directory.
2. ⚙️ **Move** the configuration files from our `configs` folder into `CoOp/configs/trainers/MPSTuning/`.
3. 💻 **Import** our class in your `main` execution file: `MPSTuning` (or `MPSTuning_ClaudeRefined`).
4. 🏃‍♂️ **Run** the training script exactly as you would train standard CoOp, just specifying the `MPSTuning` trainer and configs! 🎯

## 📝 Citation
If this work or code is helpful to your research 🌟, please consider citing us ☕:

```bibtex
@article{chen2025preserve,
  title={Preserve and Sculpt: Manifold-Aligned Fine-tuning of Vision-Language Models for Few-Shot Learning},
  author={Chen, Dexia and Zhu, Qianjie and Li, Weibing and Yu, Yue and Zhang, Tong and Wang, Ruixuan},
  journal={arXiv preprint arXiv:2508.12877},
  year={2025}
}
```

## 🙏 Acknowledgments
Our work is largely built upon [CoOp](https://github.com/KaiyangZhou/CoOp) 🔗 and [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) 🔗. We deeply appreciate the authors for their excellent work 🏆 and their invaluable contributions to the open-source community! 🌍💖
