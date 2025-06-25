import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any, Optional

# Define a target mapping for clarity if needed, though LabelEncoder handles it
TARGET_MAP = {'likely_obstructive': 0, 'likely_central': 1, 'ambiguous': 2} # Example
INV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}


def train_classification_model(
    features_df: pd.DataFrame,
    target_column: str,
    model_type: str = 'random_forest', # 'random_forest', 'logistic_regression', 'svm'
    test_size: float = 0.25,
    random_state: int = 42,
    n_folds: int = 5
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """
    Trains a classification model on the engineered features.

    Args:
        features_df (pd.DataFrame): DataFrame containing features and the target column.
                                    Assumes 'event_id' and 'event_type_original' might exist and are not features.
        target_column (str): The name of the column in features_df that contains the labels
                             (e.g., 'heuristic_label').
        model_type (str): Type of model to train.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        n_folds (int): Number of folds for cross-validation.

    Returns:
        Tuple[Any, pd.DataFrame, Dict[str, Any]]:
            - Trained model pipeline.
            - DataFrame with predictions on the test set (or full set if test_size is 0).
            - Dictionary containing evaluation metrics (accuracy, classification_report, confusion_matrix_df).
    """
    if features_df.empty:
        raise ValueError("Input features_df is empty.")
    if target_column not in features_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in features_df.")

    # Drop rows where target is NaN, as they cannot be used for training/evaluation
    features_df_cleaned = features_df.dropna(subset=[target_column])
    if features_df_cleaned.empty:
        raise ValueError(f"No valid data remaining after dropping NaNs in target column '{target_column}'.")


    # Prepare data: X (features), y (target)
    # Exclude non-feature columns like IDs or original event types not used directly as features
    cols_to_drop = ['event_id', 'event_type_original', target_column,
                    'excluded_due_to_leak_original', # Original flag, might be a feature if transformed
                    # Add any other non-feature columns that might be present
                   ]
    X = features_df_cleaned.drop(columns=[col for col in cols_to_drop if col in features_df_cleaned.columns], errors='ignore')
    y_raw = features_df_cleaned[target_column]

    # Encode string labels to numerical if they are not already
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Target classes found: {le.classes_}")
    if len(le.classes_) < 2:
        raise ValueError(f"The target column '{target_column}' must have at least two distinct classes. Found: {le.classes_}")


    # Identify numeric feature columns for imputation and scaling
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    if not numeric_features:
        raise ValueError("No numeric features found for training.")

    X_numeric = X[numeric_features]

    # Split data or use all for CV if test_size is 0
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else: # Use all data for training and evaluate via CV or on the training set itself.
        X_train, X_test, y_train, y_test = X_numeric, X_numeric, y, y # Test on training data, or rely on CV
        print("Warning: test_size is 0. Evaluating model on the training data or via Cross-Validation.")


    # Define model
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    elif model_type == 'logistic_regression':
        # Logistic Regression might benefit from more feature selection/regularization if many features
        model = LogisticRegression(random_state=random_state, solver='liblinear', class_weight='balanced', max_iter=1000)
    elif model_type == 'svm':
        model = SVC(random_state=random_state, class_weight='balanced', probability=True) # probability for predict_proba
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Create a pipeline for preprocessing (imputation, scaling) and modeling
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # Median is robust to outliers
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # --- Cross-validation ---
    cv_scores = []
    if n_folds > 1 and len(X_train) > n_folds : # Ensure enough samples for CV
        # Use StratifiedKFold for classification tasks to preserve class proportions
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        # Perform cross-validation on the training set
        # Note: pipeline handles imputation/scaling within each fold correctly
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"Cross-validation accuracy scores ({n_folds}-fold): {cv_scores}")
        print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    else:
        print("Skipping cross-validation (not enough folds or samples).")


    # --- Evaluation on the test set (or training set if test_size=0) ---
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    # Store predictions
    # Create a DataFrame for predictions, aligning with original index if possible
    # This assumes X_test maintains its original index from features_df_cleaned
    pred_df = pd.DataFrame(index=X_test.index)
    pred_df['true_label_encoded'] = y_test
    pred_df['predicted_label_encoded'] = y_pred
    pred_df['true_label'] = le.inverse_transform(y_test)
    pred_df['predicted_label'] = le.inverse_transform(y_pred)

    if y_pred_proba is not None:
        for i, class_name in enumerate(le.classes_):
            pred_df[f'proba_{class_name}'] = y_pred_proba[:, i]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    report_str = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    print("\nTest Set Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report_str)
    print("Confusion Matrix:")
    print(cm_df)

    metrics = {
        'accuracy': accuracy,
        'classification_report_dict': report_dict,
        'classification_report_str': report_str,
        'confusion_matrix_df': cm_df,
        'cv_mean_accuracy': np.mean(cv_scores) if len(cv_scores) > 0 else np.nan,
        'cv_std_accuracy': np.std(cv_scores) if len(cv_scores) > 0 else np.nan,
        'label_encoder_classes': le.classes_.tolist()
    }

    return pipeline, pred_df, metrics


if __name__ == '__main__':
    # Create a dummy features DataFrame for testing
    # This would typically come from feature_engineering.py
    num_events = 100
    num_features = 20
    rng = np.random.RandomState(42)

    data = rng.rand(num_events, num_features)
    # Make some features more discriminative for testing
    data[:, 0] *= 2 # Feature 0
    data[:, 1] -= 1 # Feature 1

    feature_names = [f'feature_{i}' for i in range(num_features)]
    dummy_features_df = pd.DataFrame(data, columns=feature_names)
    dummy_features_df['event_id'] = [f'event_{i}' for i in range(num_events)]
    dummy_features_df['event_type_original'] = rng.choice(['apnea_candidate', 'hypopnea_candidate'], size=num_events)

    # Create a dummy target column (heuristic labels)
    # Simulate some correlation with features for meaningful training
    labels = []
    for i in range(num_events):
        if data[i,0] > 1.0 and data[i,1] < -0.5:
            labels.append('likely_obstructive')
        elif data[i,0] < 0.5:
            labels.append('likely_central')
        else:
            labels.append('ambiguous') # Add a third class

    dummy_features_df['heuristic_label'] = labels

    # Test with some NaNs in features
    for _ in range(30): # Introduce 30 NaN values randomly
        row_idx = rng.randint(0, num_events)
        col_idx = rng.randint(0, num_features) # Only in numeric features
        dummy_features_df.iloc[row_idx, dummy_features_df.columns.get_loc(f'feature_{col_idx}')] = np.nan

    # Test with some NaNs in target (should be dropped)
    dummy_features_df.loc[rng.choice(dummy_features_df.index, size=5, replace=False), 'heuristic_label'] = np.nan


    print("--- Testing Model Training (Random Forest) ---")
    try:
        rf_model, rf_preds, rf_metrics = train_classification_model(
            dummy_features_df.copy(),
            target_column='heuristic_label',
            model_type='random_forest',
            test_size=0.25,
            n_folds=3 # Use 3 folds for small dummy data
        )
        print(f"Random Forest trained. Test accuracy: {rf_metrics['accuracy']:.4f}")
        print("Predictions sample:")
        print(rf_preds.head())
        assert rf_metrics['accuracy'] > 0.1 # Basic sanity check for this dummy data

        # Test if label_encoder_classes is stored
        assert 'label_encoder_classes' in rf_metrics
        print(f"Encoded classes: {rf_metrics['label_encoder_classes']}")


    except ValueError as e:
        print(f"Error during RF training test: {e}")
        # This might happen if dummy data is too small or classes are too imbalanced after NaN drop

    print("\n--- Testing Model Training (Logistic Regression) ---")
    try:
        lr_model, lr_preds, lr_metrics = train_classification_model(
            dummy_features_df.copy(),
            target_column='heuristic_label',
            model_type='logistic_regression',
            test_size=0.3,
            n_folds=3
        )
        print(f"Logistic Regression trained. Test accuracy: {lr_metrics['accuracy']:.4f}")
        assert lr_metrics['accuracy'] > 0.1
    except ValueError as e:
        print(f"Error during LR training test: {e}")

    print("\n--- Testing with insufficient classes in target ---")
    dummy_one_class_df = dummy_features_df.copy()
    # Keep only 'ambiguous' or make all one class after dropping NaNs in target
    valid_labels_idx = dummy_one_class_df['heuristic_label'].dropna().index
    dummy_one_class_df.loc[valid_labels_idx, 'heuristic_label'] = 'ambiguous' # All one class

    try:
        train_classification_model(dummy_one_class_df, 'heuristic_label')
    except ValueError as e:
        print(f"Correctly caught error for insufficient classes: {e}")
        assert "must have at least two distinct classes" in str(e)

    print("\n--- Testing with target NaNs leading to empty dataframe ---")
    dummy_all_nan_target_df = dummy_features_df.copy()
    dummy_all_nan_target_df['heuristic_label'] = np.nan
    try:
        train_classification_model(dummy_all_nan_target_df, 'heuristic_label')
    except ValueError as e:
        print(f"Correctly caught error for all NaNs in target: {e}")
        assert "No valid data remaining" in str(e)

    print("\nClassification model module tests complete.")

# Add imports for type hinting if needed at the top
from typing import Tuple, Dict, Any, Optional
