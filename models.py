import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingClassifier, VotingRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report,
                            mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR


MODEL_DIR = 'model'
DATA_FILE = 'data/dataset.csv'



CLASSIFIERS = {
    'Random Forest': RandomForestClassifier(n_estimators=5, max_depth=2, min_samples_split=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=10, max_depth=1, learning_rate=0.01, random_state=42),
    'SVM': SVC(kernel='linear', C=0.01, probability=True, random_state=42)
}

REGRESSORS = {
    'Random Forest': RandomForestRegressor(n_estimators=5, max_depth=2, min_samples_split=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=10, max_depth=1, learning_rate=0.01, random_state=42),
    'SVR': SVR(kernel='linear', C=0.001, epsilon=1.0, max_iter=5)
}

def load_data():
    df = pd.read_csv(DATA_FILE)
    train_df = df[df['train'] == 1].copy()
    test_df = df[df['train'] == 0].copy()
    
    features_to_drop = ['segment', 'Mission', 'train', 'channel', 'duration']
    
    X_train = train_df.drop(columns=features_to_drop)
    y_train_class = train_df['Mission']
    y_train_reg = train_df['duration']
    
    X_test = test_df.drop(columns=features_to_drop)
    y_test_class = test_df['Mission']
    y_test_reg = test_df['duration']
    
    return X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg, train_df, test_df

def save_model(model, filename):
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    filepath = os.path.join(MODEL_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def train_classifiers():
    X_train, X_test, y_train, y_test, _, _, _, _ = load_data()
    
    results = {}
    models = {}
    
    for name, clf in CLASSIFIERS.items():
        model_file = f'classifier_{name.replace(" ", "_").lower()}.pkl'
        loaded_model = load_model(model_file)
        
        if loaded_model is not None:
            clf = loaded_model
            print(f"Loaded existing {name} classifier")
        else:
            clf.fit(X_train, y_train)
            save_model(clf, model_file)
            print(f"Trained and saved {name} classifier")
        
        models[name] = clf
        y_pred = clf.predict(X_test)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, target_names=['Failure', 'Success'])
        }
    
    ensemble_file = 'classifier_ensemble.pkl'
    ensemble_model = load_model(ensemble_file)
    
    if ensemble_model is not None:
        ensemble_clf = ensemble_model
        print("Loaded existing ensemble classifier")
    else:



        extra_trees = ExtraTreesClassifier()
        catboost_clf = CatBoostClassifier()

        ensemble_clf = VotingClassifier(
            estimators=[('extra_trees', extra_trees), ('catboost', catboost_clf)],
            voting='soft'
            )

        ensemble_clf.fit(X_train, y_train)
        save_model(ensemble_clf, ensemble_file)
    
        
        save_model(ensemble_clf, ensemble_file)
        print("Trained and saved ensemble classifier")

    
    models['Ensemble'] = ensemble_clf
    ensemble_preds = ensemble_clf.predict(X_test)
    
    results['Ensemble'] = {
        'accuracy': accuracy_score(y_test, ensemble_preds),
        'precision': precision_score(y_test, ensemble_preds, average='weighted'),
        'recall': recall_score(y_test, ensemble_preds, average='weighted'),
        'f1': f1_score(y_test, ensemble_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, ensemble_preds),
        'classification_report': classification_report(y_test, ensemble_preds, target_names=['Failure', 'Success'])
    }
    
    return results, models

def train_regressors():
    X_train, X_test, _, _, y_train, y_test, _, _ = load_data()
    
    results = {}
    models = {}
    
    for name, reg in REGRESSORS.items():
        model_file = f'regressor_{name.replace(" ", "_").lower()}.pkl'
        loaded_model = load_model(model_file)
        
        if loaded_model is not None:
            reg = loaded_model
            print(f"Loaded existing {name} regressor")
        else:
            reg.fit(X_train, y_train)
            save_model(reg, model_file)
            print(f"Trained and saved {name} regressor")
        
        models[name] = reg
        y_pred = reg.predict(X_test)
        
        results[name] = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist()
        }
    
    ensemble_file = 'regressor_ensemble.pkl'
    ensemble_model = load_model(ensemble_file)
    
    if ensemble_model is not None:
        ensemble_reg = ensemble_model
        print("Loaded existing ensemble regressor")
    else:
        extra_trees = ExtraTreesRegressor(n_estimators=100, random_state=42)
        catboost_reg = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0, random_state=42)

        ensemble_reg = VotingRegressor(
            estimators=[('extra_trees', extra_trees), ('catboost', catboost_reg)]
        )

        ensemble_reg.fit(X_train, y_train)
        save_model(ensemble_reg, ensemble_file)
        print("Trained and saved ensemble regressor")
    
    models['Ensemble'] = ensemble_reg
    ensemble_preds = ensemble_reg.predict(X_test)
    
    results['Ensemble'] = {
        'mae': mean_absolute_error(y_test, ensemble_preds),
        'mse': mean_squared_error(y_test, ensemble_preds),
        'rmse': np.sqrt(mean_squared_error(y_test, ensemble_preds)),
        'r2': r2_score(y_test, ensemble_preds),
        'y_test': y_test.tolist(),
        'y_pred': ensemble_preds.tolist()
    }
    
    return results, models

def plot_confusion_matrix(cm, name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Failure', 'Success'], 
                yticklabels=['Failure', 'Success'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    filename = f'static/plots/cm_{name.replace(" ", "_").lower()}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()
    return filename

def plot_scatter(y_test, y_pred, name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)
    plt.xlabel('Actual Duration')
    plt.ylabel('Predicted Duration')
    plt.title(f'Actual vs Predicted - {name}')
    plt.grid(True, alpha=0.3)
    filename = f'static/plots/scatter_{name.replace(" ", "_").lower()}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=100)
    plt.close()
    return filename

def predict_single(features_dict, model_type='classification'):
    X_train, _, _, _, _, _, _, _ = load_data()
    feature_names = X_train.columns.tolist()
    
    features = [features_dict.get(col, 0) for col in feature_names]
    features_array = np.array(features).reshape(1, -1)
    
    if model_type == 'classification':
        predictions = {}
        for name in list(CLASSIFIERS.keys()) + ['Ensemble']:
            if name == 'Ensemble':
                model_file = 'classifier_ensemble.pkl'
            else:
                model_file = f'classifier_{name.replace(" ", "_").lower()}.pkl'
            
            model = load_model(model_file)
            if model:
                predictions[name] = int(model.predict(features_array)[0])
        return predictions
    else:
        predictions = {}
        for name in list(REGRESSORS.keys()) + ['Ensemble']:
            if name == 'Ensemble':
                model_file = 'regressor_ensemble.pkl'
            else:
                model_file = f'regressor_{name.replace(" ", "_").lower()}.pkl'
            
            model = load_model(model_file)
            if model:
                predictions[name] = float(model.predict(features_array)[0])
        return predictions

def predict_batch(csv_file):
    df = pd.read_csv(csv_file)
    
    features_to_drop = ['segment', 'Mission', 'train', 'channel', 'duration']
    existing_features = [col for col in features_to_drop if col in df.columns]
    X = df.drop(columns=existing_features)
    
    classification_results = []
    regression_results = []
    
    for name in list(CLASSIFIERS.keys()) + ['Ensemble']:
        if name == 'Ensemble':
            model_file = 'classifier_ensemble.pkl'
        else:
            model_file = f'classifier_{name.replace(" ", "_").lower()}.pkl'
        
        model = load_model(model_file)
        if model is not None:
            preds = model.predict(X).tolist()
            classification_results.append({
                'model': name,
                'predictions': preds
            })
    
    for name in list(REGRESSORS.keys()) + ['Ensemble']:
        if name == 'Ensemble':
            model_file = 'regressor_ensemble.pkl'
        else:
            model_file = f'regressor_{name.replace(" ", "_").lower()}.pkl'
        
        model = load_model(model_file)
        if model is not None:
            preds = model.predict(X).tolist()
            regression_results.append({
                'model': name,
                'predictions': preds
            })
    
    return classification_results, regression_results, len(X)
