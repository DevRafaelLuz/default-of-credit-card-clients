import numpy as np
import pandas as pd
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

dados = pd.read_csv('default_of_credit_card_clients.csv', sep=';')

X = dados.drop(columns=['default payment next month'])
y = dados['default payment next month']

dados_categoricos = ['SEX', 'EDUCATION', 'MARRIAGE']
dados_numericos = [
    'LIMIT_BAL', 
    'AGE', 
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

scaler = MinMaxScaler()
X_num = pd.DataFrame(scaler.fit_transform(X[dados_numericos]), columns=dados_numericos)
X_cat = pd.get_dummies(X[dados_categoricos].astype(str), prefix_sep='_', dtype=int)

X_processado = X_num.join(X_cat)

X_train, X_test, y_train, y_test = train_test_split(X_processado, y, test_size=0.3, random_state=42, stratify=y)

resampler = SMOTE(random_state=42)
X_train_b, y_train_b = resampler.fit_resample(X_train, y_train)
print(f'Frequência das classes após SMOTE: {Counter(y_train_b)}')

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_b, y_train_b)

rf_grid = {
    'n_estimators': [int(x) for x in np.linspace(10, 100, 5)],
    'criterion': ['gini', 'entropy'],
    'max_depth': [int(x) for x in np.linspace(10, 50, 5)],
    'min_samples_split': [2, 5]
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_grid,
    n_iter=10,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

rf_search.fit(X_train_b, y_train_b)
rf_best = rf_search.best_estimator_

scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']

scores_dt = cross_validate(dt, X_train_b, y_train_b, scoring=scoring, cv=5)
scores_rf = cross_validate(rf_best, X_train_b, y_train_b, scoring=scoring, cv=5)

f1_dt = scores_dt['test_f1_macro'].mean()
f1_rf = scores_rf['test_f1_macro'].mean()

print(f"\n=== SELEÇÃO DE MODELO (Média F1-Macro) ===")
print(f"Decision Tree: {f1_dt:.4f}")
print(f"Random Forest: {f1_rf:.4f}")

if f1_rf > f1_dt:
    print("-> Melhor modelo: Random Forest")
    modelo_final = rf_best
else:
    print("-> Melhor modelo: Decision Tree")
    modelo_final = dt

modelo_final.fit(X_train_b, y_train_b)

artefatos = {
    'scaler': scaler,
    'dados_treino': X_processado.columns.tolist(),
    'dados_numericos': dados_numericos,
    'dados_categoricos': dados_categoricos,
    'modelo': modelo_final
}

with open('pipeline_banco.pkl', 'wb') as f:
    pickle.dump(artefatos, f)