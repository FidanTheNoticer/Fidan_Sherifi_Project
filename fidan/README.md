# Swiss Financial Health Classification

## 📋 Description du Projet

Ce projet utilise le **Machine Learning** pour classifier la santé financière des résidents suisses en **5 catégories**:

🟢 **Très Bonne** → Score 80-100  
🟡 **Bonne** → Score 65-79  
🟠 **Moyenne** → Score 45-64  
🔴 **Mauvaise** → Score 25-44  
⚫ **Très Mauvaise** → Score 0-24  

Le projet comprend:
- ✅ **Modèles de classification** (Random Forest, Gradient Boosting, Logistic Regression)
- ✅ **Interface graphique** pour les utilisateurs finaux
- ✅ **Système de recommandations** personnalisées
- ✅ **Visualisations** (confusion matrix, feature importance)
- ✅ **Évaluation complète** (accuracy, precision, recall, F1-score)

---

## 🎯 Question de Recherche

**Peut-on prédire la classe de santé financière d'un individu (5 classes) sur la base de ses données démographiques et financières?**

---

## 📊 Dataset

**Fichier**: `dataset_sante_financiere_suisse_classification.csv`  
**Taille**: 10,000 observations × 15 features  
**Source**: Données synthétiques réalistes basées sur l'économie suisse

### Features

**Démographiques:**
- `age`: Âge de la personne (25-65 ans)
- `canton`: Canton suisse (26 cantons: ZH, GE, VD, BE, etc.)
- `situation_maritale`: célibataire, marié, divorcé, veuf
- `nombre_enfants`: 0-5 enfants
- `taux_occupation`: 50%, 80% ou 100%

**Financières:**
- `salaire_mensuel`: Salaire mensuel brut (CHF)
- `loyer_mensuel`: Loyer mensuel (CHF)
- `depenses_vitales`: Nourriture, transport, santé (CHF/mois)
- `depenses_loisirs`: Restaurants, sorties, hobbies (CHF/mois)
- `a_credit`: Oui/Non
- `montant_credit_mensuel`: Remboursement mensuel du crédit (CHF)

**Indicateurs calculés:**
- `ratio_loyer_salaire`: Loyer / Salaire × 100
- `taux_epargne`: (Salaire - Dépenses totales) / Salaire × 100
- `score_sante_financiere`: Score numérique 0-100

**Target (classification):**
- `santé_financière`: **très_mauvaise, mauvaise, moyenne, bonne, très_bonne**

### Distribution des Classes

```
très_mauvaise:  17.0%
mauvaise:       27.2%
moyenne:         7.6%
bonne:          12.1%
très_bonne:     36.0%
```

---

## 🗂️ Structure du Projet

```
financial-health-project/
├── README.md                                    # Ce fichier
├── requirements.txt                             # Dépendances Python
├── main.py                                      # Script d'entraînement
├── app_interface.py                             # Interface graphique (GUI)
│
├── src/
│   ├── __init__.py                              # Package Python
│   ├── data_loader.py                           # Chargement et preprocessing
│   ├── models.py                                # Définition des modèles
│   └── evaluation.py                            # Métriques et recommandations
│
├── data/
│   └── raw/
│       └── dataset_sante_financiere_suisse_classification.csv
│
├── models/                                      # Modèles sauvegardés (générés par main.py)
│   ├── best_model.pkl
│   ├── encoders.pkl
│   ├── feature_names.pkl
│   ├── class_names.pkl
│   ├── model_metadata.pkl
│   └── dataset_for_recommendations.pkl
│
├── results/                                     # Visualisations (générées par main.py)
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── class_distribution.png
│
└── notebooks/                                   # (optionnel) Analyses exploratoires
```

---

## ⚙️ Installation

### Prérequis
- Python 3.11+
- pip ou conda

### Étape 1: Cloner le projet

```bash
git clone <votre-repo>
cd financial-health-project
```

### Étape 2: Créer l'environnement virtuel

**Option A: Avec venv**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

**Option B: Avec conda**
```bash
conda create -n finance-health python=3.11
conda activate finance-health
```

### Étape 3: Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 4: Vérifier le dataset

Le fichier `dataset_sante_financiere_suisse_classification.csv` doit être dans `data/raw/`.

---

## 🚀 Usage

### 1. Entraîner les modèles (exécution académique)

```bash
python main.py
```

**Ce script va:**
1. ✅ Charger et préprocesser les données (10,000 obs)
2. ✅ Entraîner 3 modèles de classification
3. ✅ Évaluer les performances (accuracy, F1-score, confusion matrix)
4. ✅ Sauvegarder le meilleur modèle dans `models/`
5. ✅ Générer les visualisations dans `results/`
6. ✅ Afficher 3 exemples de recommandations

**Durée:** ~30-60 secondes

### 2. Utiliser l'interface graphique (utilisation pratique)

```bash
python app_interface.py
```

**L'application permet:**
- 📋 Saisir vos informations personnelles et financières
- 🔮 Obtenir votre classe de santé financière
- 📊 Voir les probabilités pour chaque classe
- 💡 Recevoir 3 recommandations personnalisées

---

## 🤖 Modèles de Classification

### 1. Random Forest Classifier
- **Paramètres**: n_estimators=200, max_depth=15
- **Avantages**: Robuste, gère les interactions, fournit feature importance
- **Accuracy attendue**: ~88-92%

### 2. Gradient Boosting Classifier
- **Paramètres**: n_estimators=150, learning_rate=0.1
- **Avantages**: Apprentissage séquentiel, haute performance
- **Accuracy attendue**: ~87-91%

### 3. Logistic Regression (baseline)
- **Paramètres**: Multi-class='multinomial', solver='lbfgs'
- **Avantages**: Rapide, interprétable, baseline
- **Accuracy attendue**: ~75-82%

---

## 📈 Métriques d'Évaluation

### Métriques Globales
- **Accuracy**: Pourcentage de prédictions correctes
- **Precision (weighted)**: Précision pondérée par classe
- **Recall (weighted)**: Rappel pondéré par classe
- **F1-Score (weighted)**: Moyenne harmonique de precision et recall

### Métriques Par Classe
- **Classification Report**: Precision, Recall, F1 pour chaque classe
- **Confusion Matrix**: Visualisation des erreurs de classification
- **Support**: Nombre d'observations par classe

### Cross-Validation
- **5-Fold Stratified CV**: Validation croisée stratifiée pour éviter le biais

---

## 💡 Système de Recommandations

Le système analyse **5 dimensions** financières:

1. **🏠 Logement**: Si ratio loyer/salaire > 35%
2. **🎉 Loisirs**: Si dépenses > 15% du salaire
3. **💳 Crédit**: Si remboursement > 20% du salaire
4. **💰 Épargne**: Si taux d'épargne < 10%
5. **📈 Revenus**: Si taux d'occupation < 100%

Chaque recommandation contient:
- ⚠️ **Problème identifié**
- ✅ **Action concrète**
- 📈 **Impact estimé**
- 🔴/🟡 **Niveau de priorité**

---

## 🖥️ Interface Graphique

L'application (`app_interface.py`) offre:

### Saisie
- Informations personnelles (âge, canton, situation, enfants, occupation)
- Informations financières (salaire, loyer, dépenses, crédit)

### Résultats
- 🎯 **Classe prédite** avec emoji et couleur
- 📊 **Probabilités** pour chaque classe (graphique à barres)
- 📋 **Résumé complet** de la situation
- 💡 **3 recommandations prioritaires**

---

## 📊 Visualisations Générées

### 1. Confusion Matrix (`results/confusion_matrix.png`)
Montre la performance du modèle classe par classe.

### 2. Feature Importance (`results/feature_importance.png`)
Top 10 des features les plus importantes.

### 3. Class Distribution (`results/class_distribution.png`)
Distribution des classes dans train et test sets.

---

## 🔬 Résultats Attendus

### Performance des Modèles

| Modèle              | Accuracy | F1-Score | Temps    |
|---------------------|----------|----------|----------|
| Random Forest       | ~90%     | ~0.89    | ~3-5 sec |
| Gradient Boosting   | ~89%     | ~0.88    | ~8-10 sec|
| Logistic Regression | ~78%     | ~0.77    | ~1 sec   |

### Features les Plus Importantes

1. **taux_epargne** (le plus important)
2. **ratio_loyer_salaire**
3. **salaire_mensuel**
4. **montant_credit_mensuel**
5. **depenses_loisirs**

---

## 📝 Exemples d'Utilisation

### Exemple 1: Situation Critique (⚫ très_mauvaise)

**Input:**
- Âge: 42 ans, Canton: GE, Marié, 2 enfants, 100%
- Salaire: 8000 CHF, Loyer: 3500 CHF, Crédit: 2000 CHF/mois

**Output:**
- Classe: **très_mauvaise**
- Probabilité: 85%
- Recommandations:
  1. [HAUTE] Renégocier le crédit
  2. [HAUTE] Réduire le loyer (déménagement)
  3. [HAUTE] Optimiser le budget (dépenses loisirs)

### Exemple 2: Bonne Situation (🟢 très_bonne)

**Input:**
- Âge: 35 ans, Canton: LU, Célibataire, 0 enfants, 100%
- Salaire: 7500 CHF, Loyer: 1200 CHF, Pas de crédit

**Output:**
- Classe: **très_bonne**
- Probabilité: 92%
- Recommandations: Continuer ainsi, envisager des investissements

---

## 🛠️ Commandes Utiles

```bash
# Entraîner le modèle
python main.py

# Lancer l'application GUI
python app_interface.py

# Test sur environnement vierge
python -m venv test_env
source test_env/bin/activate  # ou test_env\Scripts\activate
pip install -r requirements.txt
python main.py
python app_interface.py
```

---

## ⚠️ Limitations

1. **Données synthétiques**: Patterns réels peuvent différer
2. **Snapshot statique**: Ne capture pas l'évolution temporelle
3. **Facteurs manquants**: Actifs, dettes, éducation non inclus
4. **Classes déséquilibrées**: Distribution inégale (17%-36%)

---

## 🔮 Améliorations Futures

- [ ] Intégration de données réelles (BFS/OFS)
- [ ] Modèles de séries temporelles (évolution)
- [ ] Deep Learning (Neural Networks)
- [ ] API REST pour intégration externe
- [ ] Dashboard interactif (Streamlit/Dash)
- [ ] Export PDF des recommandations
- [ ] Multi-langues (FR/DE/IT/EN)
- [ ] Simulation "what-if" dans le GUI

---

## 📚 Dépendances

```
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
joblib >= 1.3.0
```

Voir `requirements.txt` pour la liste complète.

---

## 👥 Auteurs

Projet de Data Science - Advanced Programming 2026

---

## 📜 Licence

Usage éducatif uniquement.

---

## 🙏 Remerciements

- Office fédéral de la statistique (OFS/BFS) pour les données économiques suisses
- Scikit-learn pour les outils ML
- Communauté Python pour les bibliothèques open-source

---

**🚀 Projet prêt à l'emploi! Lancez `python main.py` puis `python app_interface.py`**
