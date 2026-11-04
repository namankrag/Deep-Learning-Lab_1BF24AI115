# Deep Learning Lab

**Student:** Naman Kumar Agrawal  
**USN:** 1BF24AI115  
**Course:** Deep Learning Lab  
**Repository:** [Deep-Learning-Lab_1BF24AI115](https://github.com/namankrag/Deep-Learning-Lab_1BF24AI115)

This repository contains Python implementations, experiments, result files, and visualizations completed as part of the Deep Learning Lab coursework.

## Repository Contents

| Folder / File | Description |
|---|---|
| [SELF LEARNING _ LAB PROGRAM_1](./SELF%20LEARNING%20_%20LAB%20PROGRAM_1/) | Self-learning lab program covering gradient-descent methods, optimizers, learning-rate schedules, regularization, early stopping, and MLP architecture comparison. |

Each lab-program folder contains its implementation and the associated documentation or results, where applicable.

## Lab Programs

### 1. Self Learning — Lab Program 1

[**Open program folder**](./SELF%20LEARNING%20_%20LAB%20PROGRAM_1/) · [**Read program documentation**](./SELF%20LEARNING%20_%20LAB%20PROGRAM_1/README.md)

This program explores optimization and generalization using PyTorch.

**Experiments and topics:**

- Batch, stochastic, and mini-batch gradient descent using a two-class `make_moons` dataset.
- Comparison of SGD with momentum, Adam, and RMSprop.
- Constant learning rate versus StepLR learning-rate scheduling.
- Fashion-MNIST image classification using a multilayer perceptron (MLP).
- Overfitting analysis using training and validation curves.
- Regularization comparison: dropout, L2 weight decay, and data augmentation.
- Early stopping based on validation loss.
- Plain MLP versus residual MLP with a skip connection.

**Dataset:** Fashion-MNIST is downloaded automatically by the program on its first run. The experiment also generates a small `make_moons` dataset for the gradient-descent demonstration.

**Recorded result:** In the committed experiment results, the residual MLP achieved **86.80% validation accuracy** and **85.35% test accuracy** on the selected Fashion-MNIST subsets. Refer to the program's [results CSV](./SELF%20LEARNING%20_%20LAB%20PROGRAM_1/fashion_mnist_results.csv) and [README](./SELF%20LEARNING%20_%20LAB%20PROGRAM_1/README.md) for the complete comparison and plots.

## Technologies Used

- **Language:** Python
- **Deep learning framework:** PyTorch and torchvision
- **Numerical computing:** NumPy
- **Visualization:** Matplotlib
- **Machine learning utilities:** scikit-learn
- **Development / execution:** Python environment, Jupyter Notebook, or terminal

## General Setup

Clone the repository:

```bash
git clone https://github.com/namankrag/Deep-Learning-Lab_1BF24AI115.git
cd Deep-Learning-Lab_1BF24AI115
```

Install the dependencies required by Self Learning — Lab Program 1:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

Move into the program folder and run its Python implementation:

```bash
cd "SELF LEARNING _ LAB PROGRAM_1"
python "SELF LEARNING _ LAB PROGRAM_1.py"
```

The program uses CUDA when a compatible GPU is available and otherwise runs on the CPU. Fashion-MNIST will be downloaded automatically when required.

> Dependencies may differ between lab programs. Check the README inside each program folder for its specific requirements and execution instructions.

## Repository Structure

```text
Deep-Learning-Lab_1BF24AI115/
├── README.md
└── SELF LEARNING _ LAB PROGRAM_1/
    ├── README.md
    ├── SELF LEARNING _ LAB PROGRAM_1.py
    ├── fashion_mnist_results.csv
    ├── baseline_overfitting_curves.png
    ├── gradient_descent_comparison.png
    ├── optimizer_and_schedule_comparison.png
    ├── plain_vs_residual_mlp.png
    └── regularization_comparison.png
```

## Purpose

This repository serves as a collection of Deep Learning Lab coursework, documenting implementations and experiments along with their results. The root README provides an overview, while the README inside each program folder contains the detailed explanation for that specific lab.

---

*Deep Learning Lab coursework — USN: 1BF24AI115.*
