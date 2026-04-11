# Anemia screening from peripheral blood smears
# Computer-aided **screening** (healthy vs anemic) from peripheral blood smear images using deep learning and handcrafted morphology/texture features. **Not a medical device** — not for diagnosis; clinical use requires validation and regulatory clearance.
## What this project does
- Loads PBS images from a fixed folder layout (main track: **AneRBC-II**).
- Trains a **ResNet18** CNN baseline and a **hybrid model** that fuses CNN embeddings with a **14-dimensional** handcrafted vector (morphology, GLCM texture, color/pallor-style scalars).
- Fits a **`StandardScaler`** on **training** handcrafted features only; validation/inference use the same scaler.
- Supports **stratified train/validation splits** (e.g. 80/20, seed 42) and saving split file lists under `data/processed/splits/` or `artifacts/splits/`.
- Optional: **Grad-CAM** visualizations and a **Streamlit** UI for upload + explanation-style outputs.
## Repository layout (typical)
- `src/` — datasets, models, training helpers, handcrafted features, Grad-CAM utilities.
- Notebooks — end-to-end exploration, training, and evaluation.

To Run this code on your device:

1. Open terminal, navigate to the path where you want to clone the project.
2. git clone https://github.com/nehalshivane04/Accelerator---Anemia-Dataset.git
3. cd Accelerator---Anemia-Dataset
4. python -m venv venv
5. venv\Scripts\activate
6. pip install torch, torchvision, numpy, pandas, Pillow, matplotlib, seaborn, scikit-learn, scikit-image, tqdm, joblib, streamlit
7. jupyter notebook
   
   Download the dataset, and replace the hardcoded paths with paths of dataset, preferably save it in the project directory itself under data/raw/
   Download dataset from here: https://kaggle.com/datasets/acdafd7c19bf90652f79e8fb8c2ee918abbac84196b059c424e72e87e4716162
