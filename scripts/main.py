#script principal - pipeline simplificado

import os
import sys
import subprocess

def run_script(script_path):
    #Ejecuta un script de Python
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error ejecutando {script_path}: {e}")
        return False

def run_pipeline():
    #Ejecuta el pipeline completo
    print("=" * 60)
    print("PIPELINE - BANK MARKETING")
    print("=" * 60)
    
    scripts = [
        ('scripts/01_data_exploration.py', 'Exploración y limpieza'),
        ('scripts/02_data_preprocessing.py', 'Preprocesamiento'),
        ('scripts/03_train_model.py', 'Entrenamiento del modelo')
    ]
    
    for i, (script_path, description) in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] {description}...")
        if not run_script(script_path):
            print(f"\n Error en el paso {i}. Pipeline detenido.")
            return False
    
    print("\n" + "=" * 60)
    print("✓ PIPELINE COMPLETADO")
    print("=" * 60)
    print("\nResultados disponibles en:")
    print("  - Modelo: models/model.pkl")
    print("  - Métricas: reports/metrics.json y reports/metrics_report.txt")
    print("  - Importancia: reports/feature_importance.csv")
    print("  - Visualizaciones: reports/*.png")
    return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)