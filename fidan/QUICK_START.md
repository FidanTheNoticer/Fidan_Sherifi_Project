# 🚀 QUICK START - Classification de Santé Financière

Guide de démarrage rapide pour lancer le projet en **5 minutes**.

---

## ⚡ Installation Rapide

### 1. Prérequis
- Python 3.11+
- pip installé

### 2. Cloner et setup (3 commandes)

```bash
# 1. Aller dans le dossier du projet
cd financial-health-project

# 2. Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OU
venv\Scripts\activate     # Windows

# 3. Installer dépendances
pip install -r requirements.txt
```

---

## 🎯 Usage Rapide

### Option A: Entraîner le modèle (pour le projet académique)

```bash
python main.py
```

**Résultats:**
- ✅ Modèles entraînés et sauvegardés dans `models/`
- ✅ Visualisations générées dans `results/`
- ✅ Métriques affichées dans la console
- ⏱️ Durée: ~30-60 secondes

### Option B: Utiliser l'interface graphique (pour tester)

```bash
# 1. Entraîner d'abord (si pas déjà fait)
python main.py

# 2. Lancer l'interface
python app_interface.py
```

**Interface:**
1. Remplir les informations personnelles
2. Remplir les informations financières
3. Cliquer sur "Classifier ma santé financière"
4. Voir la classe prédite + recommandations

---

## 📁 Structure Minimale Requise

```
financial-health-project/
├── main.py
├── app_interface.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   └── evaluation.py
└── data/
    └── raw/
        └── dataset_sante_financiere_suisse_classification.csv
```

---

## 🧪 Test Complet (5 étapes)

```bash
# 1. Activer environnement
source venv/bin/activate

# 2. Vérifier dataset
ls data/raw/dataset_sante_financiere_suisse_classification.csv

# 3. Entraîner modèles
python main.py
# → Doit afficher "✅ EXECUTION COMPLETED SUCCESSFULLY"

# 4. Vérifier fichiers générés
ls models/  # Doit contenir 6 fichiers .pkl
ls results/ # Doit contenir 3 fichiers .png

# 5. Lancer GUI
python app_interface.py
# → Interface doit s'ouvrir
```

---

## ✅ Checklist de Validation

**Avant soumission:**

- [ ] `python main.py` s'exécute sans erreur
- [ ] Dossier `models/` contient 6 fichiers .pkl
- [ ] Dossier `results/` contient 3 fichiers .png
- [ ] `python app_interface.py` lance l'interface
- [ ] Interface affiche correctement la prédiction
- [ ] README.md est complet
- [ ] Code est commenté et lisible
- [ ] requirements.txt est à jour

---

## 🐛 Troubleshooting Rapide

### Erreur: "No module named 'src'"
```bash
# Solution: Vous n'êtes pas dans le bon dossier
cd financial-health-project
python main.py
```

### Erreur: "No such file or directory: 'data/raw/...'"
```bash
# Solution: Dataset manquant
# Vérifier que le fichier CSV est bien dans data/raw/
ls data/raw/
```

### Erreur: "Modèle non trouvé" dans l'interface
```bash
# Solution: Entraîner d'abord le modèle
python main.py
# Puis relancer l'interface
python app_interface.py
```

### Interface ne s'affiche pas
```bash
# Solution: Vérifier que tkinter est installé
python -m tkinter
# Si erreur, installer: sudo apt-get install python3-tk (Linux)
```

---

## 📊 Résultats Attendus

### Console (main.py)

```
================================================================================
SWISS FINANCIAL HEALTH CLASSIFICATION - ML PROJECT
================================================================================

[1/6] Loading and preprocessing data...
Dataset loaded: 10000 rows × 15 columns
Target classes: ['bonne' 'mauvaise' 'moyenne' 'très_bonne' 'très_mauvaise']
...

🏆 BEST MODEL: Random Forest
   Accuracy: 0.9012 | F1-Score: 0.8954

✅ EXECUTION COMPLETED SUCCESSFULLY
```

### Interface Graphique (app_interface.py)

- Formulaire avec tous les champs
- Bouton "Classifier ma santé financière"
- Résultats avec:
  - Classe prédite (emoji + couleur)
  - Probabilités par classe
  - 3 recommandations personnalisées

---

## ⏱️ Timing

| Tâche                    | Durée      |
|--------------------------|------------|
| Installation             | 2-3 min    |
| Entraînement (main.py)   | 30-60 sec  |
| Test GUI                 | 1-2 min    |
| **TOTAL**                | **5 min**  |

---

## 🎓 Pour le Rendu Académique

**Fichiers à inclure:**

1. **Code source** (tous les .py)
2. **Dataset** (.csv)
3. **README.md** (ce fichier)
4. **requirements.txt**
5. **Rapport PDF** (à rédiger séparément)

**Ne PAS inclure:**
- Dossier `models/` (fichiers .pkl trop lourds)
- Dossier `results/` (images générées)
- Dossier `venv/` (environnement virtuel)

---

## 🚀 Commandes Essentielles

```bash
# Setup
pip install -r requirements.txt

# Entraînement
python main.py

# Interface
python app_interface.py

# Nettoyage
rm -rf models/*.pkl results/*.png
```

---

**🎉 Vous êtes prêt! Lancez `python main.py` pour commencer.**
