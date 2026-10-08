# Self-Supervised Ocular Disease Representation Learning

Recovered implementation of an undergraduate research/TCC project investigating whether representations learned by **autoencoders from unlabeled medical images** can support downstream ocular-disease classification.

## Project scope

The recovered code contains the main experimental ideas:

- image preprocessing at **512×512**;
- extraction of the **V channel from HSV**;
- a five-stage convolutional autoencoder;
- a convolutional variational autoencoder (VAE);
- a residual autoencoder;
- additional sparse and denoising autoencoder experiments in the recovered notebook;
- reconstruction metrics including PSNR and SSIM;
- frozen encoders used as feature extractors;
- downstream classification with SVM, KNN and Random Forest.

The associated paper also reports a **cross-dataset generalization experiment**. The complete source code for that evaluation was not clearly present in the recovered notebook, so this repository does **not** pretend that portion was recovered.

## Recovered results

The saved notebook outputs include, on the recovered 70/15/15 split of the four-class ocular dataset:

| Encoder / representation | Classifier | Test accuracy |
| --- | --- | ---: |
| Convolutional encoder | SVM | 84.83% |
| Convolutional encoder | Random Forest | 80.73% |
| Convolutional encoder | KNN | 65.72% |
| VAE encoder | SVM | 81.83% |
| Residual encoder | SVM | 84.04% |
| Denoising encoder | SVM | 76.62% |
| Sparse encoder | SVM | 56.56% |

The paper reports that the cross-dataset experiment suffered a substantial performance drop, highlighting domain shift rather than hiding it.

## Authorship

### Implementation

The experimental code in this repository was implemented by **Gustavo Fraiz Badaro Barbosa**.

### Research paper

The associated paper, *Aprendizado auto-supervisionado Utilizando Autoencoders com Foco em Doenças Oculares*, was written collaboratively by:

- Gustavo Albiero
- Gustavo Fraiz Badaro Barbosa
- Renan Antonio Hammerschmidt Krefta
- Victor Silva Camargo
- Vinícius Silva Camargo

The paper itself is not redistributed here by default; this repository focuses on the recovered code and reproducibility material.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── models.py
├── data/
│   └── README.md
└── models/
    └── README.md
```

## Reproducibility note

This is a **recovered academic codebase**, not a rewritten project presented as if it had originally been structured this way. The core architectures were extracted from the recovered Colab notebook into `src/models.py` for easier inspection. The dataset files and trained weights are not committed.

No license is included by default.
