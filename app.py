from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
from functools import wraps
from database import init_db, register_user, verify_user
from models import (train_classifiers, train_regressors, load_data, 
                    plot_confusion_matrix, plot_scatter, predict_single, predict_batch,
                    CLASSIFIERS, REGRESSORS)
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'satellite-telemetry-secret-key-2024')

init_db()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        mobile = request.form.get('mobile')
        email = request.form.get('email')
        address = request.form.get('address')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        success, message = register_user(name, mobile, email, address, password)
        if success:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash(f'Registration failed: {message}', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        success, user = verify_user(email, password)
        if success:
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            flash(f'Welcome back, {user["name"]}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/eda')
@login_required
def eda():
    df = pd.read_csv('data/dataset.csv')
    
    stats = {
        'total_records': len(df),
        'train_records': len(df[df['train'] == 1]),
        'test_records': len(df[df['train'] == 0]),
        'mission_success': len(df[df['Mission'] == 1]),
        'mission_failure': len(df[df['Mission'] == 0]),
        'features': len(df.columns) - 5,
        'avg_duration': round(df['duration'].mean(), 2),
        'min_duration': df['duration'].min(),
        'max_duration': df['duration'].max()
    }
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[numerical_cols].describe().round(4).to_html(classes='table table-striped table-bordered')
    
    return render_template('eda.html', stats=stats, summary=summary)

@app.route('/classification')
@login_required
def classification():
    print("Training classifiers...")
    results, models = train_classifiers()
    
    for name, result in results.items():
        if 'confusion_matrix' in result:
            cm_file = plot_confusion_matrix(result['confusion_matrix'], name)
            result['cm_plot'] = cm_file.replace('static/', '')
    
    classifier_names = list(CLASSIFIERS.keys()) + ['Ensemble']
    
    return render_template('classification.html', results=results, classifier_names=classifier_names)

@app.route('/regression')
@login_required
def regression():
    print("Training regressors...")
    results, models = train_regressors()
    
    for name, result in results.items():
        if 'y_test' in result and 'y_pred' in result:
            scatter_file = plot_scatter(result['y_test'], result['y_pred'], name)
            result['scatter_plot'] = scatter_file.replace('static/', '')
    
    regressor_names = list(REGRESSORS.keys()) + ['Ensemble']
    
    return render_template('regression.html', results=results, regressor_names=regressor_names)

@app.route('/comparison')
@login_required
def comparison():
    print("Generating comparison...")
    clf_results, _ = train_classifiers()
    reg_results, _ = train_regressors()
    
    classifier_names = list(CLASSIFIERS.keys()) + ['Ensemble']
    regressor_names = list(REGRESSORS.keys()) + ['Ensemble']
    
    return render_template('comparison.html', 
                         clf_results=clf_results, 
                         reg_results=reg_results,
                         classifier_names=classifier_names,
                         regressor_names=regressor_names)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    X_train, _, _, _, _, _, _, _ = load_data()
    feature_names = X_train.columns.tolist()
    
    single_result = None
    batch_results = None
    
    if request.method == 'POST':
        if 'single_predict' in request.form:
            features = {}
            for feature in feature_names:
                value = request.form.get(feature, '0')
                try:
                    features[feature] = float(value)
                except:
                    features[feature] = 0.0
            
            class_preds = predict_single(features, 'classification')
            reg_preds = predict_single(features, 'regression')
            
            single_result = {
                'classification': class_preds,
                'regression': reg_preds
            }
        
        elif 'batch_file' in request.files:
            file = request.files['batch_file']
            if file and file.filename.endswith('.csv'):
                filepath = 'data/uploaded_test.csv'
                file.save(filepath)
                
                class_results, reg_results, num_rows = predict_batch(filepath)
                batch_results = {
                    'classification': class_results,
                    'regression': reg_results,
                    'num_rows': num_rows
                }
    
    return render_template('predict.html', 
                         feature_names=feature_names,
                         single_result=single_result,
                         batch_results=batch_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
