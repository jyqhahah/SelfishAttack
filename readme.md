# Competitive Advantage Attacks to Decentralized Federated Learning

This repository contains the official code for our **NeurIPS 2025** paper:  
**[“Competitive Advantage Attacks to Decentralized Federated Learning”](https://arxiv.org/pdf/2310.13862)**  

---

## 🚀 Environment Setup

We recommend using **conda** to manage environments.

Create environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate selfish
````

---

## 🏋️ Run

To train the model and launch the SelfishAttack, use the following command (example for **FedAvg**):

```bash
python3 main.py --dataset=cifar10 --model=cnnc --bias=0.7 --seed=1 --epochs=700 \
        --lr=0.0001 --batchsize=128 --nworkers=20 --nbyz=6 --cmax=6 --aggregation=fedavg \
        --gpu=3 --attack_epoch=50 --local_round=3 --alpha=0.0 --epsilon=0.1
````

### Attack Settings

* **FedAvg-based Attack**:
  `--aggregation fedavg --alpha 0.0`
* **Median-based Attack**:
  `--aggregation median --alpha 0.5`
* **Trimmed-mean-based Attack**:
  `--aggregation trim --alpha 1.0`


---

## 📖 Citation

If you use this code in your work, please kindly cite the following paper:

```bibtex
@inproceedings{jia2025competitive,
  title={Competitive Advantage Attacks to Decentralized Federated Learning},
  author={Jia, Yuqi and Fang, Minghong and Gong, Neil},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```