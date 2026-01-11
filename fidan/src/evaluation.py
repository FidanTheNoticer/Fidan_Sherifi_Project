"""
Model evaluation and recommendation system for classification.
"""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model(model, X_test, y_test, model_name, class_names):
    """
    Evaluate classification model performance on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name (str): Name of the model
        class_names (list): List of class names

    Returns:
        dict: Dictionary containing all metrics
    """
    y_pred = model.predict(X_test)

    # Métriques globales
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1-Score (weighted): {f1:.4f}")

    # Rapport de classification détaillé
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def plot_confusion_matrix(cm, class_names, model_name, save_path='results/confusion_matrix.png'):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names (list): List of class names
        model_name (str): Name of the model
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))

    # Normaliser la matrice de confusion (en pourcentage)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Créer le heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title(f'{model_name}: Matrice de Confusion', fontsize=14, fontweight='bold')
    plt.ylabel('Classe Réelle', fontsize=12)
    plt.xlabel('Classe Prédite', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Matrice de confusion sauvegardée: {save_path}")
    plt.close()


def plot_feature_importance(feature_importance, top_n=10, save_path='results/feature_importance.png'):
    """
    Plot feature importance.

    Args:
        feature_importance (list): List of (feature_name, importance) tuples
        top_n (int): Number of top features to display
        save_path (str): Path to save the plot
    """
    if feature_importance is None:
        print("\n⚠️  Feature importance not available for this model")
        return

    # Get top N features
    top_features = feature_importance[:top_n]
    features, importances = zip(*top_features)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), importances, color='steelblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Features les Plus Importantes', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance sauvegardée: {save_path}")
    plt.close()


def plot_class_distribution(y_train, y_test, class_names, save_path='results/class_distribution.png'):
    """
    Plot class distribution in train and test sets.

    Args:
        y_train: Training target
        y_test: Test target
        class_names (list): List of class names
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Train set distribution
    train_counts = y_train.value_counts().reindex(class_names)
    ax1.bar(range(len(class_names)), train_counts.values, color='steelblue')
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_ylabel('Nombre d\'observations')
    ax1.set_title('Distribution des Classes - Train Set')
    ax1.grid(axis='y', alpha=0.3)

    # Test set distribution
    test_counts = y_test.value_counts().reindex(class_names)
    ax2.bar(range(len(class_names)), test_counts.values, color='coral')
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.set_ylabel('Nombre d\'observations')
    ax2.set_title('Distribution des Classes - Test Set')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Distribution des classes sauvegardée: {save_path}")
    plt.close()


def generate_recommendations(row, df_complet, top_n=3):
    """
    Generate personalized financial recommendations based on classification.

    Args:
        row (pd.Series): Single observation with financial data
        df_complet (pd.DataFrame): Complete dataset for comparisons
        top_n (int): Number of recommendations to return

    Returns:
        list: List of recommendation dictionaries
    """
    recommandations = []

    # Déterminer la classe actuelle
    classe_actuelle = row.get('santé_financière', 'moyenne')

    # 1. ANALYSE DU RATIO LOYER/SALAIRE
    if row['ratio_loyer_salaire'] > 35:
        canton_similar = df_complet[df_complet['canton'] == row['canton']]
        loyer_moyen_canton = canton_similar['loyer_mensuel'].median()

        if row['loyer_mensuel'] > loyer_moyen_canton * 1.2:
            economie = row['loyer_mensuel'] - row['salaire_mensuel'] * 0.30
            recommandations.append({
                'categorie': 'Logement',
                'priorite': 'HAUTE',
                'probleme': f"Loyer élevé: {row['loyer_mensuel']:.0f} CHF ({row['ratio_loyer_salaire']:.1f}% du salaire, idéal: <30%)",
                'action': f"Réduire le loyer à {row['salaire_mensuel']*0.30:.0f} CHF économiserait {economie:.0f} CHF/mois",
                'impact': 'Passage potentiel à une meilleure classe'
            })

        # Suggestion de cantons moins chers
        cantons_accessibles = df_complet.groupby('canton')['loyer_mensuel'].median().sort_values().head(5)
        economie_canton = row['loyer_mensuel'] - cantons_accessibles.values[0]
        recommandations.append({
            'categorie': 'Relocalisation',
            'priorite': 'MOYENNE',
            'probleme': f"Canton {row['canton']}: loyers élevés",
            'action': f"Cantons abordables: {', '.join(cantons_accessibles.index[:3])} (économie: {economie_canton:.0f} CHF/mois)",
            'impact': 'Amélioration de la classe financière'
        })

    # 2. ANALYSE DES DÉPENSES LOISIRS
    ratio_loisirs = (row['depenses_loisirs'] / row['salaire_mensuel']) * 100
    canton_similar = df_complet[df_complet['canton'] == row['canton']]
    loisirs_moyen_canton = (canton_similar['depenses_loisirs'] / canton_similar['salaire_mensuel'] * 100).median()

    if ratio_loisirs > 15:
        economie = row['depenses_loisirs'] * 0.3
        recommandations.append({
            'categorie': 'Loisirs',
            'priorite': 'MOYENNE',
            'probleme': f"Dépenses loisirs: {row['depenses_loisirs']:.0f} CHF ({ratio_loisirs:.1f}% vs {loisirs_moyen_canton:.1f}% en moyenne)",
            'action': f"Réduire de 30% économiserait {economie:.0f} CHF/mois",
            'impact': 'Amélioration du taux d\'épargne'
        })

    # 3. ANALYSE DU CRÉDIT
    if row['a_credit'] == 'oui':
        ratio_credit = (row['montant_credit_mensuel'] / row['salaire_mensuel']) * 100
        if ratio_credit > 20:
            recommandations.append({
                'categorie': 'Crédit',
                'priorite': 'HAUTE',
                'probleme': f"Crédit lourd: {row['montant_credit_mensuel']:.0f} CHF ({ratio_credit:.1f}% du salaire)",
                'action': "Renégocier les conditions ou consolider les crédits",
                'impact': 'Impact majeur sur la classe financière'
            })

    # 4. ANALYSE DU TAUX D'ÉPARGNE
    if row['taux_epargne'] < 10:
        deficit = abs(row['salaire_mensuel'] * 0.10 - 
                      (row['salaire_mensuel'] - row['loyer_mensuel'] - 
                       row['depenses_vitales'] - row['depenses_loisirs'] - 
                       row['montant_credit_mensuel']))
        recommandations.append({
            'categorie': 'Épargne',
            'priorite': 'HAUTE',
            'probleme': f"Épargne insuffisante: {row['taux_epargne']:.1f}% (objectif: >10%)",
            'action': f"Réduire les dépenses de {deficit:.0f} CHF/mois pour atteindre 10% d\'épargne",
            'impact': 'Essentiel pour améliorer la classe'
        })

    # 5. SUGGESTION D'AUGMENTATION DE REVENUS
    if row['taux_occupation'] < 100:
        augmentation = row['salaire_mensuel'] * ((100 - row['taux_occupation']) / row['taux_occupation'])
        recommandations.append({
            'categorie': 'Revenus',
            'priorite': 'MOYENNE',
            'probleme': f"Taux d\'occupation: {row['taux_occupation']}%",
            'action': f"Passer à 100% générerait {augmentation:.0f} CHF/mois supplémentaires",
            'impact': 'Amélioration significative possible'
        })

    # Trier par priorité
    ordre_priorite = {'HAUTE': 0, 'MOYENNE': 1, 'BASSE': 2}
    recommandations.sort(key=lambda x: ordre_priorite[x['priorite']])

    return recommandations[:top_n]


def display_recommendations(recommendations):
    """
    Display recommendations in a formatted way.

    Args:
        recommendations (list): List of recommendation dictionaries
    """
    print(f"\n{'='*80}")
    print(f"💡 RECOMMANDATIONS PERSONNALISÉES ({len(recommendations)})")
    print(f"{'='*80}")

    for i, reco in enumerate(recommendations, 1):
        print(f"\n{i}. [{reco['priorite']}] {reco['categorie']}")
        print(f"   ⚠️  {reco['probleme']}")
        print(f"   ✅ {reco['action']}")
        print(f"   📈 Impact: {reco['impact']}")
