# %%
import numpy as np
import pandas as pd 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier


# %%
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, roc_curve, auc, confusion_matrix

def plot_confusion_matrix(model, X, y, model_name):
    y_pred = cross_val_predict(model, X, y, cv=5)
    cm = confusion_matrix(y, y_pred)
    
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    # plt.savefig(f'confusion_matrix_{model_name}.png')
    # plt.close()
    fig.show()
    return fig

# %%
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, cross_validate

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

def find_best_model(X, y):
    # Step 1: Normalize the data
    scaler = MinMaxScaler()          
    X_scaled = scaler.fit_transform(X)
    
    # Aditional Step 1: Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, shuffle=True, random_state=42)

    models = {
        'SVC': SVC(),
        'NuSVC': NuSVC(),
        'Linear SVC': LinearSVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'MLP Classifier': MLPClassifier(),
        'Gaussian NB': GaussianNB(),
        'Bernoulli NB': BernoulliNB(),
        'K-Neighbors': KNeighborsClassifier(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
        'Perceptron': Perceptron(),
        'SGD Classifier': SGDClassifier(),
        'Ridge Classifier': RidgeClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Passive Aggressive': PassiveAggressiveClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Bagging': BaggingClassifier(),
        'Voting': VotingClassifier(estimators=[
            ('rf', RandomForestClassifier()),
            ('dt', DecisionTreeClassifier())
        ], voting='hard'),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'Extra Trees': ExtraTreesClassifier(),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, objective='binary:logistic')
    }
    
    results = {}
    fold_scores = {}
    skf = StratifiedKFold(n_splits=5)
    
    for name, model in models.items():
        scores = cross_validate(model, X_scaled, y, cv=skf, scoring={
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'specificity': make_scorer(specificity_score)
        })
        ## later i see all the scores separate from k-fold
        results[name] = {
            'Accuracy': scores['test_accuracy'].mean(),
            'Precision': scores['test_precision'].mean(),
            'Recall (Sensitivity)': scores['test_recall'].mean(),
            'Specificity': scores['test_specificity'].mean(),
            'F1-Score': scores['test_f1'].mean()
        }
        fold_scores[name] = scores['test_accuracy'] * 100 # Convert accuracy to percentage

    # Step 2: Print the results in a presentable manner
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.sort_values(by='Accuracy', ascending=False)
    #print("Cross-Validation Results (5-Fold):")
    #print(results_df.to_string())
    
    # Aditional Step 2: Test the model's accuracy on the test set
    # best_model_name = results_df.idxmax().values[0]
    # best_model = models[best_model_name]
    # best_model.fit(X_train, y_train)

    # y_test_pred = best_model.predict(X_test)
    # test_accuracy = accuracy_score(y_test, y_test_pred)

    # print(f"\nBest Model: {best_model_name}")
    # print(f"Test Set Accuracy: {test_accuracy:.4f}")

    # return(y_test, y_test_pred)
    
    # Step 3: Plot the top 5 models
    top_5_models = results_df.head(5).index
    line_styles = ['-', '--', '-.', ':', (0, (5, 10))]  # Different line styles for each model
    
    kfold_fig = plt.figure(figsize=(15, 8))
    for i, model_name in enumerate(top_5_models):
        scores = fold_scores[model_name]
        folds = list(range(1, 6)) + [6]  # Adding 6 for the mean
        scores_with_mean = list(scores) + [scores.mean()]
        plt.plot(folds, scores_with_mean, marker='o', linestyle=line_styles[i % len(line_styles)], label=model_name)
    
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.xticks([1, 2, 3, 4, 5, 6], ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean'])
    plt.legend()
    kfold_fig.show()

    # Step 4: Plot confusion matrix of top model

    best_model_name = results_df.index[0]
    best_model = models[best_model_name]

    confusion_fig = plot_confusion_matrix(best_model, X_scaled, y, best_model_name)

    return results_df, confusion_fig, kfold_fig

