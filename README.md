# FruitVision — Projet CC3 Deep Learning
**Auteur :** Ali Ennadafy  
**Formation :** TS Intelligence Artificielle — 2025/2026  
**Formateur :** Mr. SABER
Application Streamlit comparant trois modèles de vision fine-tunés sur un dataset personnalisé de fruits (Roboflow) :
- **YOLOv8n** — détection d'objets
- **ViT-base-patch16-224** — classification d'images
- **SAM-vit-base** — segmentation (fine-tuning du décodeur)
---
## Structure du projet
Project_DL_Final/ ├── app.py # Application Streamlit ├── requirements.txt ├── README.md ├── REPORT.md # Rapport comparatif ├── data/fruit_dataset/ # Dataset YOLO (train/valid/test) ├── models/ │ ├── yolo_finetuned/weights/best.pt │ ├── vit_finetuned/ │ └── sam_finetuned/sam_decoder_best.pth ├── outputs/ │ ├── metrics/ # metrics_yolo.json, metrics_vit.json, metrics_sam.json │ └── curves/ # training_curves_*.png └── src/ ├── resplit_dataset.py # Re-split 60/20/20 (seed=42) ├── train_yolo.py ├── train_vit.py ├── train_sam.py ├── export_yolo_artifacts.py ├── inference.py └── utils/ ├── paths.py └── dataset_loader.py # Chargement unifié YOLO / ViT / SAM

---
## Installation
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
Fine-tuning
Important : après le re-split 60/20/20, relancer l'entraînement des trois modèles.

YOLO (détection)
cd src
python train_yolo.py
python export_yolo_artifacts.py
ViT (classification)
cd src
python train_vit.py
Les labels YOLO sont convertis automatiquement en classes d'image (classe majoritaire par image).

SAM (segmentation)
cd src
python train_sam.py
Fine-tuning parameter-efficient : seul le mask decoder est entraîné.

Tout entraîner d'un coup (Windows)
train_all.bat
Lancement de l'application
Depuis la racine du projet :

streamlit run app.py
L'application permet de :

Uploader une image (JPG/PNG)
Choisir un ou plusieurs modèles
Visualiser détections, classifications ou masques
Comparer les métriques dans l'onglet Comparaison
Inférence CLI
cd src
python inference.py --model all --image ../data/fruit_dataset/test/images/IMG.jpg
Dataset
Source : Roboflow — Fruit Detection
Total : 2194 images · 6 classes
Split : 60 / 20 / 20 (train: 1316, valid: 438, test: 440)
Re-split : aléatoire seed=42 via src/resplit_dataset.py (conforme au cahier des charges)
Classes
Apple (0), Banana (1), Orange (2), Grape (3), Mango (4), Cherry (5)

Formats par modèle
Modèle	Format utilisé
YOLO
images/ + labels/*.txt — format YOLO standard (class cx cy w h normalisés)
ViT
Labels dérivés des annotations YOLO — classe majoritaire par image
SAM
Bbox YOLO converties en pixels comme prompt + masque rectangle comme ground-truth
Structure
data/fruit_dataset/
├── data.yaml
├── train/images/  +  train/labels/
├── valid/images/  +  valid/labels/
└── test/images/   +  test/labels/
Le chargement unifié est dans src/utils/dataset_loader.py.

Métriques exportées
Chaque script génère :

outputs/metrics/metrics_<modele>.json
outputs/curves/training_curves_<modele>.png
L'application Streamlit lit ces fichiers depuis outputs/metrics/ et outputs/curves/ pour l'onglet comparaison.

Auteur
Ali Ennadafy — Projet réalisé dans le cadre du contrôle continu CC3 (Deep Learning).
