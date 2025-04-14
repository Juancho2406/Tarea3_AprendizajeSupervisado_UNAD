import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def cargar_datos():
    """Carga y realiza la limpieza inicial del dataset"""
    # Cargar el dataset
    df = pd.read_csv('../datasets/winequality-red.csv', sep=';')
    return df

def analisis_exploratorio(df):
    """Realiza el análisis exploratorio de datos completo"""
    
    # 1. Estadísticas básicas
    print("Estadísticas descriptivas:")
    print(df.describe())
    
    # 2. Distribución de la calidad del vino
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='quality')
    plt.title('Distribución de Calidad del Vino')
    plt.xlabel('Calidad')
    plt.ylabel('Cantidad')
    plt.show()
    
    # 3. Distribución de variables fisicoquímicas
    variables = df.columns.drop('quality')
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.ravel()
    
    for idx, col in enumerate(variables):
        sns.histplot(data=df, x=col, kde=True, ax=axes[idx])
        axes[idx].set_title(f'Distribución de {col}')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Boxplots por calidad
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    axes = axes.ravel()
    
    for idx, col in enumerate(variables):
        sns.boxplot(data=df, x='quality', y=col, ax=axes[idx])
        axes[idx].set_title(f'{col} vs Calidad')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Matriz de correlación
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación')
    plt.show()
    
    # 6. Pairplot de variables más importantes
    important_vars = ['alcohol', 'volatile acidity', 'sulphates', 'quality']
    sns.pairplot(df[important_vars], hue='quality', diag_kind='hist')
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