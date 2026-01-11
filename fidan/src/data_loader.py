"""
Data loading and preprocessing for financial health classification.
⚠️ taux_epargne ET ratio_loyer_salaire sont EXCLUS des features.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset(filepath='data/raw/dataset_sante_financiere_suisse_classification.csv'):
    """
    Load the Swiss financial health classification dataset.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nTarget classes: {df['santé_financière'].unique()}")
    return df


def encode_categorical_features(df):
    """
    Encode categorical variables to numeric for ML models.

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        tuple: (encoded_df, dict of label encoders)
    """
    df_encoded = df.copy()

    # Initialize label encoders
    encoders = {
        'canton': LabelEncoder(),
        'situation_maritale': LabelEncoder(),
        'a_credit': LabelEncoder()
    }

    # Encode categorical features
    df_encoded['canton_encoded'] = encoders['canton'].fit_transform(df['canton'])
    df_encoded['situation_encoded'] = encoders['situation_maritale'].fit_transform(df['situation_maritale'])
    df_encoded['credit_encoded'] = encoders['a_credit'].fit_transform(df['a_credit'])

    print("\nCategorical features encoded:")
    print(f"  - canton: {len(encoders['canton'].classes_)} unique values")
    print(f"  - situation_maritale: {len(encoders['situation_maritale'].classes_)} unique values")
    print(f"  - a_credit: {len(encoders['a_credit'].classes_)} unique values")

    return df_encoded, encoders


def prepare_features_target(df_encoded):
    """
    Prepare feature matrix X and target variable y for classification.
    ⚠️ taux_epargne ET ratio_loyer_salaire sont EXCLUS (variables calculées).

    Args:
        df_encoded (pd.DataFrame): Dataframe with encoded features

    Returns:
        tuple: (X, y, feature_names, class_names)
    """
    # Features SANS taux_epargne NI ratio_loyer_salaire (11 features)
    feature_names = [
        'age',
        'canton_encoded',
        'situation_encoded',
        'nombre_enfants',
        'taux_occupation',
        'salaire_mensuel',
        'loyer_mensuel',
        'depenses_vitales',
        'depenses_loisirs',
        'credit_encoded',
        'montant_credit_mensuel'
    ]

    # Vérifier que toutes les features existent dans le dataframe
    missing_features = [f for f in feature_names if f not in df_encoded.columns]
    if missing_features:
        print(f"\n⚠️  WARNING: Missing features: {missing_features}")
        print(f"Available columns: {list(df_encoded.columns)}")
        raise KeyError(f"Features not found in dataframe: {missing_features}")

    X = df_encoded[feature_names]
    y = df_encoded['santé_financière']

    # Obtenir les noms des classes
    class_names = sorted(y.unique())

    print(f"\nFeatures matrix: {X.shape}")
    print(f"Target variable: {y.shape}")
    print(f"Classes: {class_names}")
    print(f"\n⚠️  Note: taux_epargne ET ratio_loyer_salaire EXCLUS ({len(feature_names)} features)")

    return X, y, feature_names, class_names


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.

    Args:
        X: Feature matrix
        y: Target variable
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nData split completed (stratified):")
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # Afficher la distribution des classes
    print(f"\nClass distribution in train set:")
    train_dist = y_train.value_counts(normalize=True).sort_index() * 100
    for classe, pct in train_dist.items():
        print(f"  {classe}: {pct:.1f}%")

    return X_train, X_test, y_train, y_test


def load_and_preprocess(filepath='data/raw/dataset_sante_financiere_suisse_classification.csv', 
                        test_size=0.2, 
                        random_state=42):
    """
    Complete pipeline: load, encode, split data for classification.

    Args:
        filepath (str): Path to dataset
        test_size (float): Proportion of test set
        random_state (int): Random seed

    Returns:
        dict: Dictionary containing all processed data and metadata
    """
    # Load dataset
    df = load_dataset(filepath)

    # Encode categorical features
    df_encoded, encoders = encode_categorical_features(df)

    # Prepare features and target
    X, y, feature_names, class_names = prepare_features_target(df_encoded)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    return {
        'df_original': df,
        'df_encoded': df_encoded,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'class_names': class_names,
        'encoders': encoders
    }
