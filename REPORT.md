# Rapport comparatif — FruitVision

**Auteur :** Ali Ennadafy  
**Date :** 2026  
**Projet :** CC3 — Apprentissage profond

---

## 1. Objectif

Fine-tuner trois modèles de vision (YOLOv8, ViT, SAM) sur un dataset personnalisé de fruits et comparer leurs performances via une application Streamlit locale.

---

## 2. Dataset

| Split | Images | %    |
|-------|--------|------|
| Train | 1316   | 60%  |
| Valid | 438    | 20%  |
| Test  | 440    | 20%  |

**Classes :** Apple, Banana, Orange, Grape, Mango, Cherry  
**Format :** YOLO (images + labels .txt)

---

## 3. Tableau comparatif

| Métrique                 | YOLOv8    | ViT            | SAM          |
| ------------------------ | --------- | -------------- | ------------ |
| Tâche                    | Détection | Classification | Segmentation |
| Precision                | 0.5975    | 0.5736         | —            |
| Recall                   | 0.3424    | 0.5767         | —            |
| F1-score                 | 0.4353    | 0.5641         | —            |
| mAP@50                   | 0.3957    | —              | —            |
| Mean IoU                 | —         | —              | 0.7645       |
| Latence (ms/image)       | 766.66    | 306.43         | 13766.89     |
| Taille du modèle (MB)    | 5.93      | 327.34         | 357.69       |
| Temps d'entraînement (s) | 49837.6   | 9295.9         | 291.0        |
| Époques entraînées       | 15        | 8              | 1            |


> Les valeurs exactes sont dans `outputs/metrics/` après entraînement.

---

## 4. Courbes d'apprentissage

Les courbes sont sauvegardées dans `outputs/curves/` :
- `training_curves_yolo.png`
- `training_curves_vit.png`
- `training_curves_sam.png`

---

## Analyse comparative

Les résultats montrent que chaque modèle répond à un besoin différent.

### YOLOv8

YOLOv8 a obtenu un mAP@50 de 39.57 % avec une taille de modèle très faible (5.93 MB). Il est adapté aux applications temps réel grâce à son faible encombrement mémoire et sa capacité à détecter plusieurs objets simultanément.

### ViT

Le modèle ViT a obtenu le meilleur score de classification avec un F1-score de 56.41 %. Il est particulièrement adapté lorsque l'objectif est uniquement d'identifier la classe dominante de l'image sans localisation spatiale.

### SAM

SAM a atteint un Mean IoU de 76.45 %, ce qui représente la meilleure précision spatiale parmi les trois approches. Cependant, il présente la latence la plus élevée et le modèle le plus volumineux.

### Recommandation

Pour une application de détection en temps réel, YOLOv8 constitue le meilleur compromis. Pour la classification d'images, ViT est le plus adapté. Pour les tâches nécessitant une segmentation précise au niveau pixel, SAM reste la meilleure solution malgré son coût de calcul plus important.


### 5. Tests sur 10 images

Les modèles ont été testés sur 10 images non vues du jeu de test.
Les résultats obtenus sont cohérents avec les métriques calculées sur l'ensemble du dataset.

## 6. Conclusion

Les trois modèles sont complémentaires. YOLOv8 est le plus adapté pour une chaîne de production rapide. ViT convient quand seule la classe dominante importe. SAM est recommandé pour des applications nécessitant des masques précis.

**Développé par Ali Ennadafy** — exécution 100 % locale (PyTorch, Ultralytics, HuggingFace).
