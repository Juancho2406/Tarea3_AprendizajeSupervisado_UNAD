import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración de visualización
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def cargar_datos():
    """Carga y realiza la limpieza inicial del dataset"""
    # Cargar el dataset
    df = pd.read_csv('../datasets/heart_cleveland_upload.csv')
    return df

def analisis_exploratorio(df):
    """Realiza el análisis exploratorio de datos completo"""
    
    # 1. Estadísticas básicas
    print("Estadísticas descriptivas:")
    print(df.describe())
    
    # 2. Distribución de la variable objetivo
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='condition')
    plt.title('Distribución de Enfermedades Cardíacas')
    plt.xlabel('Presencia de Enfermedad')
    plt.ylabel('Cantidad')
    plt.show()
    
    # 3. Distribución de variables numéricas
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        sns.histplot(data=df, x=col, kde=True, ax=axes[idx])
        axes[idx].set_title(f'Distribución de {col}')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Análisis por variables categóricas
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(categorical_cols):
        sns.countplot(data=df, x=col, ax=axes[idx])
        axes[idx].set_title(f'Distribución de {col}')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Relación entre variables numéricas y objetivo
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        sns.boxplot(data=df, x='condition', y=col, ax=axes[idx])
        axes[idx].set_title(f'{col} vs Condición')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Matriz de correlación
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación')
    plt.show()

def main():
    """Función principal que ejecuta el análisis"""
    try:
        # Cargar y preparar datos
        df = cargar_datos()
        print("Datos cargados exitosamente!")
        print(f"Dimensiones del dataset: {df.shape}")
        
        # Realizar análisis exploratorio
        analisis_exploratorio(df)
        
    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")

if __name__ == "__main__":
    main()