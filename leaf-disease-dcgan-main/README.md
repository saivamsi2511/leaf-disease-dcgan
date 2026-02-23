# 🌱 Leaf Disease DCGAN
Deep Convolutional Generative Adversarial Network (DCGAN) for generating synthetic crop leaf disease images to improve training datasets for plant disease classification systems.

---

## 📌 Project Overview

Farmers and agri-tech systems increasingly rely on image-based disease detection apps. However, real-world datasets often suffer from:

- Severe class imbalance
- Limited samples of rare diseases
- Noisy and inconsistent field images
- High data collection costs

This project uses a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic synthetic diseased leaf images that can be used to:

- Augment training datasets
- Improve classification accuracy
- Reduce data collection effort
- Support research and education

---

## 🎯 Project Objectives

- Generate realistic synthetic crop leaf disease images
- Improve data availability for rare disease classes
- Build a full end-to-end GAN pipeline
- Provide a web interface for image generation

---

## 🧠 Model Architecture

### Generator
- Input: 100-dimensional latent noise vector
- Architecture:
  - Fully convolutional transpose layers
  - Batch normalization
  - ReLU activations
- Output: **64×64 RGB synthetic leaf image**
- Final activation: **Tanh**

### Discriminator
- Input: 64×64 RGB image
- Architecture:
  - Convolutional layers
  - Batch normalization
  - LeakyReLU activations
- Output: Probability of real vs fake image
- Final activation: **Sigmoid**

---

## 📂 Project Structure

leaf-disease-dcgan/
│
├── src/
│ ├── generator.py # Generator model
│ ├── discriminator.py # Discriminator model
│ ├── train_dcgan.py # GAN training script
│ ├── data_loader.py # Dataset loader
│ ├── prepare_data.py # Data splitting
│ ├── inference.py # Image generation script
│ └── app_leaf_gan.py # Streamlit web app
│
├── configs/
│ └── data_config.yaml # Dataset configuration
│
├── checkpoints/ # Saved model weights
├── samples/ # Generated images per epoch
├── figures/ # Evaluation plots
└── README.md


---

## ⚙️ Tech Stack

- **Python**
- **PyTorch**
- Streamlit
- NumPy
- Pandas
- Matplotlib
- PIL (Pillow)

---

## 📊 Dataset

Dataset used:

**PlantVillage Crop Disease Dataset**

Classes used in this project:
- Tomato Healthy
- Tomato Early Blight
- Tomato Late Blight

Images were:
- Resized to 64×64
- Normalized to [-1, 1]
- Split into train/validation/test sets

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/navadeep20/leaf-disease-dcgan.git
cd leaf-disease-dcgan
2. Create a virtual environment
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install torch torchvision matplotlib numpy pandas pillow tqdm pyyaml streamlit
4. Train the DCGAN
python src/train_dcgan.py
This will:

Train the generator and discriminator

Save checkpoints

Save generated samples per epoch

5. Generate synthetic images (inference)
python src/inference.py
Output will be saved in:

generated/synthetic_leaves.png
6. Run the web app
streamlit run src/app_leaf_gan.py
Features:

Select number of images

Generate synthetic leaves

Visualize outputs in real time

📈 Training Results
Model trained for 20 epochs

Generator learned leaf-like shapes and disease textures

Visual improvement observed across epochs

Synthetic images suitable for dataset augmentation

Evaluation outputs:

Loss curves

Epoch comparison grids

🌍 Real-World Applications
Agri-tech disease detection apps

Government agricultural advisory tools

Seed and pesticide research

Academic plant pathology studies

AI training data augmentation

🔮 Future Improvements
Conditional GAN for multi-disease generation

FID and Inception Score evaluation

Integration with disease classification models

Deployment as a cloud-based API

Mobile-friendly interface

👤 Author
Batch-8, GAN for Images
B.Tech CSE 
GitHub: https://github.com/navadeep20

