import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_preprocess_data():
    """Carga y preprocesa los datos para el entrenamiento"""
    # Cargar datos
    df = pd.read_csv('../datasets/winequality-red.csv', sep=';')
    
    # Separar features y target
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, X.columns

def train_model(X, y):
    """Entrena el modelo de árbol de decisión"""
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, feature_names):
    """Evalúa el modelo y muestra métricas"""
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nMétricas de Evaluación:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.show()
    
    # Visualizar árbol
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, 
             class_names=[str(i) for i in model.classes_],
             filled=True, rounded=True)
    plt.title('Árbol de Decisión')
    plt.show()
    
    # Importancia de características
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=True)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('Importancia de Características')
    plt.xlabel('Importancia')
    plt.show()
    
    return accuracy

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
        joblib.dump(model, 'wine_quality_model.joblib')
        joblib.dump(scaler, 'wine_quality_scaler.joblib')
        print("Modelo guardado exitosamente!")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")

if __name__ == "__main__":
    main()