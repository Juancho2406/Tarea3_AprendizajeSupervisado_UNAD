import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_preprocess_data():
    """Carga y preprocesa los datos para el entrenamiento"""
    # Cargar datos
    df = pd.read_csv('../datasets/heart_cleveland_upload.csv')
    
    # Separar features y target
    X = df.drop('condition', axis=1)
    y = df['condition']
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns

def train_model(X, y):
    """Entrena el modelo de regresión logística"""
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, feature_names):
    """Evalúa el modelo y muestra métricas"""
    # Hacer predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nMétricas de Evaluación:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.show()
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    
    # Importancia de características
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(model.coef_[0])
    })
    feature_importance = feature_importance.sort_values('importance', ascending=True)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('Importancia de Características')
    plt.xlabel('Importancia Absoluta')
    plt.show()
    
    return accuracy, precision, recall, f1

def main():
    """Función principal que ejecuta el entrenamiento"""
    try:
        print("1. Cargando y preprocesando datos...")
        X, y, scaler, feature_names = load_and_preprocess_data()
        
        print("2. Entrenando modelo...")
        model, X_train, X_test, y_train, y_test = train_model(X, y)
        
        print("3. Evaluando modelo...")
        evaluate_model(model, X_test, y_test, feature_names)
        
        print("\n4. Guardando modelo...")
        joblib.dump(model, 'heart_disease_model.joblib')
        joblib.dump(scaler, 'heart_disease_scaler.joblib')
        print("Modelo guardado exitosamente!")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")

if __name__ == "__main__":
    main()