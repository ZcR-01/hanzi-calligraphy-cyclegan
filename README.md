# Hanzi Calligraphy Style Transfer with CycleGAN

This project explores unpaired image-to-image translation to generate Chinese calligraphy characters from standard printed fonts using CycleGAN.  
The goal is to learn a calligraphic style from examples and apply it to clean font characters, helping users understand and compare different calligraphy styles.

This project was developed as part of a Master’s degree practical assignment.

## Project Overview

- **Task**: Font → Calligraphy style transfer
- **Model**: CycleGAN (unpaired GAN)
- **Framework**: PyTorch
- **Input**: Standard Chinese Song style font (TTF rendered to PNG)
- **Output**: Calligraphy style character images
- **Training setup**: GPU (JupyterHub on Kubernetes)

The project focuses on data preparation, experimental design, training, and qualitative evaluation.

## Motivation

Chinese calligraphy exhibits strong stylistic variation while preserving strict character structure.  
This makes it a suitable application for unpaired image translation, where exact pixel level correspondence between domains does not exist.

The objective is to transfer _style_ (brush strokes, ink texture, stroke variation) while preserving _content_ (character structure).

## Dataset

### Domain A — Standard Font

- Generated from a Song style TTF font
- 1000 common Chinese characters
- Rendered as 256×256 grayscale PNG images
- Clean and consistent structure

### Domain B — Calligraphy

- Calligraphy images of the same character set
- Unpaired (no one-to-one correspondence with Domain A)
- High stylistic variability (stroke width, ink density, texture)

## Dataset Structure

The dataset must follow the CycleGAN folder convention:

```text
dataset/
├── trainA/   # Song font characters
├── trainB/   # Calligraphy characters
├── testA/    # Unseen font characters
└── testB/    # Optional
```

## Model Architecture

CycleGAN consists of:

- Two generators:
  - G_A: Font → Calligraphy
  - G_B: Calligraphy → Font

- Two discriminators:
  - D_A: Discriminator for Calligraphy domain
  - D_B: Discriminator for Font domain

- Cycle-consistency loss to preserve character structure

The model learns a distribution level mapping between domains rather than paired correspondences.

## Dependencies

This project relies on the official PyTorch implementation of CycleGAN:

🔗 https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

The model code is not included in this repository.
Users must clone the original repository to train or test the model.

## ⚙️ Setup Instructions

### 1. Clone the CycleGAN repository

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate pytorch-img2img
```

### 3. Prepare the dataset

If you want to generate your own dataset, requeriments in pyproject.toml:

```
uv run ../scripts/generate_data.py
or
python ../scripts/generate_data.py
```

Place the dataset inside the CycleGAN repository once you have it ready:

```
pytorch-CycleGAN-and-pix2pix/
└── dataset/
    ├── trainA/
    ├── trainB/
    ├── testA/
    └── testB/
```

### 4. Create the environment

```bash
conda env create -f environment.yml
conda activate pytorch-img2img
```

### Training

```bash
python train.py \
    --dataroot ./dataset \
    --name hanzi_style \
    --model cycle_gan \
    --use_wandb
```

#### Training Details

| Parameter     | Value   |
| ------------- | ------- |
| Image size    | 256×256 |
| Batch size    | 1       |
| Epochs        | 200     |
| Identity loss | Enabled |

### Testing & Results

```bash
python test.py \
    --dataroot ./dataset \
    --name hanzi_style \
    --model cycle_gan \
```

Results are saved to `checkpoints/hanzi_style/test_latest/`

An `index.html` file is generated for qualitative inspection:

- **real_A**: Input font character
- **fake_B**: Generated calligraphy (main result)
- **rec_A**: Reconstruction (cycle consistency)

## Example Results

The model preserves character structure while successfully transferring calligraphic style features such as variable stroke width and brush-like edges.
