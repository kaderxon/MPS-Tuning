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

### 🔧 Dassl Dual-View Patch

MPS-Tuning requires **two differently augmented views** of each input image (as described in the paper). The standard Dassl data loader does not support this out of the box for single-view datasets, so you need to apply the following patch to your Dassl installation. 🩹

**Step 1: Modify `SimpleTrainer` in `dassl.engine.trainer`** 🏗️

Locate the `SimpleTrainer` class and update the `build_data_loader` method to expose the dual-view loader:

```python
def build_data_loader(self):
    dm = DataManager(self.cfg)
    self.train_loader_x = dm.train_loader_x
    self.train_loader_u = dm.train_loader_u
    self.val_loader = dm.val_loader
    self.test_loader = dm.test_loader
    self.train_loader_x_2view = dm.train_loader_x_2view  # ← add this
    self.num_classes = dm.num_classes
    self.num_source_domains = dm.num_source_domains
    self.lab2cname = dm.lab2cname
    self.dm = dm
​```

**Step 2: Update `DataManager` in `dassl.data.data_manager`** 📦

Inside the `__init__` method of `DataManager`, initialize the dual-view loader by adding:

```python
self.train_loader_x_2view = build_data_loader(
    cfg,
    sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
    data_source=dataset.train_x,
    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
    n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
    n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
    tfm=tfm_train,
    is_train=True,
    dataset_wrapper=DatasetWrapper_2view  # ← uses the new wrapper below
)
​```

**Step 3: Add `DatasetWrapper_2view` to `dassl.data.data_manager`** 🖼️🖼️

This wrapper ensures that every `__getitem__` call returns **two independently augmented versions** of the same image:

```python
class DatasetWrapper_2view(DatasetWrapper):
    def __init__(self, cfg, data_source, transform=None, is_train=False):
        super().__init__(cfg, data_source, transform, is_train)

    def __getitem__(self, idx):
        item = self.data_source[idx]
        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }
        img0 = read_image(item.impath)
        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img  = self._transform_image(self.transform, img0)
                img1 = self._transform_image(self.transform, img0)
                output["img"]  = img
                output["img1"] = img1
        else:
            output["img"] = img0
        output["img0"] = self.pure_to_tensor(img0)
        return output

    def _transform_image(self, tfm, img0):
        img_list = []
        for k in range(self.k_tfm):
            img_list.append(tfm(img0))
        img = img_list
        if len(img) == 1:
            img = img[0]
        return img
​```

After applying these three steps, your trainer will correctly generate two augmented views per sample for single-view datasets. 🎉 If you run into any issues, please don't hesitate to open an issue — we're happy to help! 🤝

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
