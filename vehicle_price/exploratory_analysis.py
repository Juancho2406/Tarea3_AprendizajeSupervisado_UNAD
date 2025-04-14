import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Update the visualization configuration
plt.style.use('seaborn-v0_8')  # Using the updated style name
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def cargar_datos():
    """Carga y realiza la limpieza inicial del dataset"""
    # Cargar el dataset
    df = pd.read_csv('../datasets/CAR DETAILS FROM CAR DEKHO.csv')
    
    # Crear columna de antigüedad del vehículo
    año_actual = datetime.now().year
    df['vehicle_age'] = año_actual - df['year']
    
    # Eliminar columna year original
    df = df.drop('year', axis=1)
    
    # Filtrar outliers
    df = df[df['selling_price'] <= 5000000]
    df = df[df['km_driven'] <= 300000]
    
    # Transformaciones logarítmicas
    df['log_price'] = np.log1p(df['selling_price'])
    df['log_km'] = np.log1p(df['km_driven'])
    
    return df

def analisis_exploratorio(df):
    """Realiza el análisis exploratorio de datos completo"""
    
    # 1. Estadísticas básicas
    print("Estadísticas descriptivas:")
    print(df.describe())
    
    # 2. Distribución de precios
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['selling_price'], kde=True)
    plt.title('Distribución de Precios')
    plt.xlabel('Precio')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['log_price'], kde=True)
    plt.title('Distribución de Log-Precios')
    plt.xlabel('Log-Precio')
    
    plt.tight_layout()
    plt.show()
    
    # 3. Relación entre kilometraje y precio
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='km_driven', y='selling_price', alpha=0.5)
    plt.title('Kilometraje vs Precio')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='log_km', y='log_price', alpha=0.5)
    plt.title('Log-Kilometraje vs Log-Precio')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Análisis por variables categóricas
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.boxplot(data=df, x='fuel', y='selling_price', ax=axes[0,0])
    axes[0,0].set_title('Precios por Tipo de Combustible')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df, x='seller_type', y='selling_price', ax=axes[0,1])
    axes[0,1].set_title('Precios por Tipo de Vendedor')
    
    sns.boxplot(data=df, x='transmission', y='selling_price', ax=axes[1,0])
    axes[1,0].set_title('Precios por Tipo de Transmisión')
    
    sns.boxplot(data=df, x='owner', y='selling_price', ax=axes[1,1])
    axes[1,1].set_title('Precios por Tipo de Propietario')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Análisis de antigüedad
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['vehicle_age'], kde=True)
    plt.title('Distribución de Antigüedad')
    plt.xlabel('Años')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='vehicle_age', y='selling_price')
    plt.title('Antigüedad vs Precio')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Matriz de correlación
    plt.figure(figsize=(10, 8))
    numeric_cols = ['selling_price', 'km_driven', 'vehicle_age', 'log_price', 'log_km']
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
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