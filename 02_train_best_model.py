from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier
from Codes.Model_architectures_tran import simple_lstm, attention_lstm_model, non_recurrent_attention_model, transformer_model
from Codes.embedding_modules import doc2vec_dm, doc2vec_dbow, word2vec_cbow, word2vec_sg, fasttext_sg, fasttext_cbow
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
import joblib


def run_end_to_end(data_path, save_path='trained_model.h5', test_size=0.2, random_state=42):
    
    data = pd.read_csv(data_path)
    # Define parameters
    parameters_one_vs_rest = {"vr__estimator__n_estimators": [100], 
                              "vr__estimator__class_weight": ["balanced"]}    
    
    # Get order of classes
    order = list(data["high_level_substr"].value_counts().index)        
    
    # Prepare label encoder
    le = LabelEncoder()
    le.fit(data[["high_level_substr"]].values.reshape(-1,1).ravel())    
    joblib.dump(le, 'label_encoder.pkl')
    
    # Simple train/test split by group
    X_train, X_test, y_train, y_test = train_test_split(
        data["sig_gene_seq"].values,
        data["high_level_substr"].values,
        test_size=test_size,
        random_state=random_state,
        stratify=data["high_level_substr"].values
    )
    
    # Create and train pipeline
    clf_one_vs_rest = Pipeline([
        ('vectorizer', CountVectorizer(
            tokenizer=lambda x: str(x).replace("|", ",").split(','), 
            lowercase=False
        )), 
        ('vr', OneVsRestClassifier(BalancedRandomForestClassifier(n_jobs=-1)))
    ])
    
    # Perform grid search
    gs_one_vs_rest = GridSearchCV(
        clf_one_vs_rest, 
        parameters_one_vs_rest, 
        cv=5, 
        n_jobs=7, 
        scoring="balanced_accuracy", 
        verbose=0
    )
    
    # Fit the model
    gs_one_vs_rest.fit(X_train, y_train)
    
    # Get best parameters and make predictions
    best_params = gs_one_vs_rest.best_params_
    y_test_pred = gs_one_vs_rest.predict(X_test)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_test_pred, labels=order, normalize='true')
    overall_acc = accuracy_score(y_test, y_test_pred)
    avg_class_acc = np.mean(np.diag(cm))
    
    # Create report
    report = pd.DataFrame(classification_report(y_test, y_test_pred, labels=order, output_dict=True)).iloc[:3, :len(order)]
    overall_report = report.mean(1)
    
    # Create confusion matrix dataframe
    df_cm = pd.DataFrame(cm, index=order, columns=order)
    
    # Create plots
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(df_cm, annot=True, annot_kws={"fontsize":12, "weight":"bold"})
    plt.title("Confusion matrix for the BOW BRF model", fontsize=20, weight="bold")
    plt.xlabel("Predicted Label", weight="bold", fontsize=20)
    plt.ylabel("True Label", weight="bold", fontsize=20)
    plt.xticks(weight="bold", fontsize=15, rotation=90)
    plt.yticks(weight="bold", fontsize=15, rotation=0)
    
    # Create classification report plot
    fig2 = plt.figure(figsize=(10, 10))
    sns.heatmap(report, annot=True)
    plt.title("Classification Report", fontsize=20)
    plt.ylabel("Metric Name", fontsize=20)
    plt.xlabel("Substrate", fontsize=20)
    plt.xticks(weight="bold", fontsize=15)
    plt.yticks(weight="bold", fontsize=15, rotation=0)
    
    # Save the best model
    joblib.dump(gs_one_vs_rest.best_estimator_, save_path)
    
    return overall_acc, avg_class_acc, best_params, gs_one_vs_rest.best_estimator_, fig, fig2, report, overall_report

accuracy, class_accuracy, best_params, best_model, confusion_matrix_plot, report_plot, report, overall_metrics = run_end_to_end(
    data_path='data/Train_data.csv',  # Your input data
    save_path='best_model.joblib',
    test_size=0.2  # Size of test set (20%)
)