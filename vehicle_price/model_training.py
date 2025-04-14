# Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_and_preprocess_data():
    """Carga y preprocesa los datos para el entrenamiento"""
    # Cargar datos
    df = pd.read_csv('../datasets/CAR DETAILS FROM CAR DEKHO.csv')
    
    # Crear columna de antigüedad
    año_actual = datetime.now().year
    df['vehicle_age'] = año_actual - df['year']
    df = df.drop('year', axis=1)
    
    # Filtrar outliers
    df = df[df['selling_price'] <= 5000000]
    df = df[df['km_driven'] <= 300000]
    
    # Transformaciones logarítmicas
    df['log_price'] = np.log1p(df['selling_price'])
    df['log_km'] = np.log1p(df['km_driven'])
    
    return df

def prepare_features(df):
    """Prepara las características para el modelo"""
    # Codificar variables categóricas
    le = LabelEncoder()
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    
    for col in categorical_cols:
        df[col + '_encoded'] = le.fit_transform(df[col])
    
    # Preparar features
    features = ['vehicle_age', 'log_km'] + [col + '_encoded' for col in categorical_cols]
    X = df[features]
    y = df['log_price']
    
    return X, y

def train_model(X, y):
    """Entrena el modelo usando GridSearchCV"""
    # Split de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Definir parámetros para GridSearch
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Entrenar modelo con GridSearch
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y muestra métricas"""
    # Hacer predicciones
    y_pred_log = model.predict(X_test)
    
    # Convertir predicciones a escala original
    y_pred = np.expm1(y_pred_log)
    y_test_original = np.expm1(y_test)
    
    # Calcular métricas
    mae = mean_absolute_error(y_test_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    r2 = r2_score(y_test_original, y_pred)
    
    print("\nMétricas de Evaluación:")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return mae, rmse, r2

def main():
    """Función principal que ejecuta el entrenamiento"""
    try:
        print("1. Cargando y preprocesando datos...")
        df = load_and_preprocess_data()
        
        print("2. Preparando características...")
        X, y = prepare_features(df)
        
        print("3. Entrenando modelo...")
        grid_search, X_test, y_test = train_model(X, y)
        
        print("\nMejores parámetros encontrados:")
        print(grid_search.best_params_)
        
        print("\n4. Evaluando modelo...")
        evaluate_model(grid_search.best_estimator_, X_test, y_test)
        
        print("\n5. Guardando modelo...")
        joblib.dump(grid_search.best_estimator_, 'vehicle_model.joblib')
        print("Modelo guardado exitosamente!")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")

if __name__ == "__main__":
    main()