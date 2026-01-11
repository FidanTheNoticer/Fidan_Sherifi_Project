"""
Main script for training and evaluating financial health classification models.
⚠️ taux_epargne ET ratio_loyer_salaire sont EXCLUS des features (variables calculées).
"""
import os
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

from src.data_loader import load_and_preprocess
from src.models import (train_random_forest, train_gradient_boosting, 
                        train_logistic_regression, get_feature_importance)
from src.evaluation import (evaluate_model, plot_confusion_matrix, 
                            plot_feature_importance, plot_class_distribution)
import joblib
import numpy as np


def main():
    print("="*80)
    print("SWISS FINANCIAL HEALTH CLASSIFICATION - ML PROJECT")
    print("⚠️  Without taux_epargne & ratio_loyer_salaire (11 base features)")
    print("="*80)

    # 1. CHARGEMENT ET PREPROCESSING
    print("\n[1/6] Loading and preprocessing data...")
    print("-"*80)
    data = load_and_preprocess(
        filepath='data/raw/dataset_sante_financiere_suisse_classification.csv',
        test_size=0.2,
        random_state=42
    )

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    class_names = data['class_names']
    encoders = data['encoders']
    df_original = data['df_original']

    # 2. ENTRAÎNEMENT DES MODÈLES
    print("\n[2/6] Training models...")
    print("-"*80)

    print("\n  → Training Random Forest Classifier...")
    rf_model = train_random_forest(X_train, y_train, n_estimators=200, max_depth=15)

    print("  → Training Gradient Boosting Classifier...")
    gb_model = train_gradient_boosting(X_train, y_train, n_estimators=150, learning_rate=0.1)

    print("  → Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)

    # 3. ÉVALUATION
    print("\n[3/6] Evaluating models...")
    print("-"*80)

    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest", class_names)
    gb_results = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting", class_names)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression", class_names)

    # 4. SÉLECTION DU MEILLEUR MODÈLE
    print("\n[4/6] Selecting best model...")
    print("-"*80)

    models_comparison = [
        (rf_model, rf_results, "Random Forest"),
        (gb_model, gb_results, "Gradient Boosting"),
        (lr_model, lr_results, "Logistic Regression")
    ]

    # Trier par F1-score
    models_comparison.sort(key=lambda x: x[1]['f1_score'], reverse=True)
    best_model, best_results, best_model_name = models_comparison[0]

    print(f"\n🏆 Best model: {best_model_name}")
    print(f"   Accuracy: {best_results['accuracy']:.4f} ({best_results['accuracy']*100:.2f}%)")
    print(f"   F1-Score: {best_results['f1_score']:.4f}")
    print(f"   Precision: {best_results['precision']:.4f}")
    print(f"   Recall: {best_results['recall']:.4f}")

    # 5. VISUALISATIONS ET SAUVEGARDE
    print("\n[5/6] Generating visualizations and saving models...")
    print("-"*80)

    # Plot confusion matrix
    plot_confusion_matrix(
        best_results['confusion_matrix'], 
        class_names, 
        best_model_name,
        'results/confusion_matrix.png'
    )

    # Plot feature importance
    feature_importance = get_feature_importance(best_model, feature_names)
    if feature_importance:
        print(f"\nTop 10 Important Features:")
        for i, (feat, imp) in enumerate(feature_importance[:10], 1):
            print(f"  {i:2d}. {feat:<30}: {imp:.4f} ({imp*100:.2f}%)")

        plot_feature_importance(feature_importance, top_n=10, save_path='results/feature_importance.png')

    # Plot class distribution
    plot_class_distribution(y_train, y_test, class_names, 'results/class_distribution.png')

    # Sauvegarder le meilleur modèle et les encodeurs
    joblib.dump(best_model, 'models/best_classifier.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    joblib.dump(class_names, 'models/class_names.pkl')

    print(f"\n✓ Model saved: models/best_classifier.pkl")
    print(f"✓ Encoders saved: models/encoders.pkl")
    print(f"✓ Metadata saved")

    # 6. EXEMPLES DE PRÉDICTIONS
    print("\n[6/6] Testing predictions on sample profiles...")
    print("-"*80)

    # Sélectionner 3 exemples de chaque classe
    for classe in class_names[:3]:  # Afficher 3 classes seulement
        samples = df_original[df_original['santé_financière'] == classe].head(1)

        if len(samples) > 0:
            profile = samples.iloc[0]

            # Calculer ratio_loyer_salaire ET taux_epargne EN INTERNE (pas dans le dataset)
            ratio_loyer_salaire = (profile['loyer_mensuel'] / profile['salaire_mensuel'] * 100) if profile['salaire_mensuel'] > 0 else 0

            depenses_totales = profile['loyer_mensuel'] + profile['depenses_vitales'] + \
                              profile['depenses_loisirs'] + profile['montant_credit_mensuel']
            epargne_mensuelle = profile['salaire_mensuel'] - depenses_totales
            taux_epargne = (epargne_mensuelle / profile['salaire_mensuel'] * 100) if profile['salaire_mensuel'] > 0 else 0

            print(f"\nClasse réelle: {classe}")
            print(f"Age: {profile['age']} ans | Canton: {profile['canton']} | "
                  f"{profile['situation_maritale']} | {profile['nombre_enfants']} enfant(s)")
            print(f"Salaire: {profile['salaire_mensuel']:.0f} CHF | "
                  f"Loyer: {profile['loyer_mensuel']:.0f} CHF ({ratio_loyer_salaire:.1f}%)")
            print(f"Dépenses: vitales={profile['depenses_vitales']:.0f} CHF, "
                  f"loisirs={profile['depenses_loisirs']:.0f} CHF")
            print(f"Crédit: {profile['a_credit']} ({profile['montant_credit_mensuel']:.0f} CHF)")
            print(f"Épargne: {taux_epargne:.1f}% ({epargne_mensuelle:.0f} CHF/mois)")

    # RÉSUMÉ FINAL
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)

    print(f"\n📊 Final Results:")
    print(f"   Model: {best_model_name}")
    print(f"   Accuracy: {best_results['accuracy']*100:.2f}%")
    print(f"   F1-Score: {best_results['f1_score']:.4f}")
    print(f"   Features used: {len(feature_names)} (11 base features)")

    print(f"\n📁 Generated files:")
    print(f"   - results/confusion_matrix.png")
    print(f"   - results/feature_importance.png")
    print(f"   - results/class_distribution.png")
    print(f"   - models/best_classifier.pkl")
    print(f"   - models/encoders.pkl")

    print("\n🚀 Next step: Launch the GUI")
    print("   python app_interface.py")
    print("="*80)


if __name__ == '__main__':
    main()
