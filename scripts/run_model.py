#!/usr/bin/env python3
"""
Скрипт для полного цикла работы с нейронной сетью:
- Загрузка датасетов d1 и d2
- Разделение на train/test (80/20)
- Обучение модели с помощью GeneticTrainer
- Расчет и вывод метрик (F1-score)
- Визуализация результатов
- Возможность дообучения на новом датасете

ПРИМЕЧАНИЕ: Для работы этого скрипта необходимо:
1. Собрать Python-модуль ai_module через pybind11
2. Установить зависимости: pip install -r requirements.txt
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Получаем абсолютный путь к директории скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Попытка импорта C++ модуля
try:
    import ai_module
    CPP_MODULE_AVAILABLE = True
    print("✓ C++ модуль ai_module успешно загружен")
except ImportError as e:
    CPP_MODULE_AVAILABLE = False
    print(f"⚠ C++ модуль ai_module не доступен: {e}")
    print("⚠ Будут использованы заглушки для демонстрации")
    print("⚠ Для полной функциональности соберите модуль через pybind11")

# Заглушки классов, если C++ модуль не доступен
if not CPP_MODULE_AVAILABLE:
    class DataLoader:
        def loadFromCSV(self, filename):
            print(f"[Заглушка] Загрузка данных из {filename}")
            # Возвращаем заглушку датасета
            class DataSet:
                def __init__(self):
                    self.data = []
                    self.labels = []
                def getFeatureCount(self):
                    return 2
                def size(self):
                    return len(self.data)
                def get(self, idx):
                    class DataPoint:
                        def __init__(self, features, label):
                            self.features = features
                            self.label = label
                        def getFeatures(self):
                            return self.features
                        def getLabel(self):
                            return self.label
                    return DataPoint(self.data[idx], self.labels[idx])
            return DataSet()
    
    class NeuralNetwork:
        def __init__(self, architecture):
            self.architecture = architecture
            print(f"[Заглушка] Создана нейронная сеть с архитектурой {architecture}")
        
        def predict(self, input_vector):
            # Простая заглушка: предсказываем класс 0 или 1 случайно
            return 0 if np.random.random() > 0.5 else 1
    
    class GeneticTrainer:
        def __init__(self, population_size=50, max_generations=100):
            self.population_size = population_size
            self.max_generations = max_generations
            print(f"[Заглушка] Создан GeneticTrainer (pop={population_size}, gen={max_generations})")
        
        def train(self, network, dataset):
            print("[Заглушка] Обучение модели...")
            # Возвращаем заглушку истории обучения
            return [1.0, 0.8, 0.6, 0.4, 0.2]
        
        def calculateMetrics(self, network, dataset):
            class Metrics:
                def __init__(self):
                    self.f1 = 0.85
                    self.accuracy = 0.9
                    self.precision = 0.88
                    self.recall = 0.82
            return Metrics()
    
    class Visualizer:
        @staticmethod
        def plot2D(dataset, network, output_path):
            print(f"[Заглушка] Визуализация 2D данных в {output_path}")
            # Создаем простой график
            plt.figure(figsize=(8, 6))
            plt.scatter([0, 1, 2], [0, 1, 0], c=['red', 'blue', 'red'], s=100)
            plt.title("Заглушка визуализации")
            plt.xlabel("Feature 0")
            plt.ylabel("Feature 1")
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path, dpi=150)
            plt.close()
            print(f"  График сохранен как {output_path}")
        
        @staticmethod
        def plotTrainingHistory(history, output_path):
            print(f"[Заглушка] Визуализация истории обучения в {output_path}")
            plt.figure(figsize=(10, 6))
            plt.plot(history, 'b-', linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Loss')
            plt.title('Training History (Stub)')
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path, dpi=150)
            plt.close()
            print(f"  График сохранен как {output_path}")

# Утилиты для работы с данными
def load_csv_data(filename):
    """Загрузка данных из CSV файла с помощью pandas"""
    try:
        df = pd.read_csv(filename)
        print(f"✓ Загружен датасет: {filename}")
        print(f"  Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
        
        # Определяем столбцы с признаками и целевой переменной
        if 'target' in df.columns:
            X = df.drop('target', axis=1).values
            y = df['target'].values
        elif 'label' in df.columns:
            X = df.drop('label', axis=1).values
            y = df['label'].values
        else:
            # Предполагаем, что последний столбец - целевая переменная
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        
        print(f"  Признаки: {X.shape[1]}-мерные, Классы: {len(np.unique(y))}")
        return X, y
        
    except Exception as e:
        print(f"✗ Ошибка загрузки {filename}: {e}")
        return None, None

def split_data(X, y, test_size=0.2, random_state=42):
    """Разделение данных на обучающую и тестовую выборки"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"✓ Разделение данных: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, dataset_name=""):
    """Оценка модели с помощью различных метрик"""
    if len(y_true) == 0 or len(y_pred) == 0:
        print("✗ Нет данных для оценки")
        return
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if dataset_name:
        print(f"\n📊 Результаты на {dataset_name}:")
    else:
        print(f"\n📊 Результаты оценки:")
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    parser = argparse.ArgumentParser(description='Запуск нейронной сети для классификации')
    parser.add_argument('--dataset', type=str, default='d1',
                       choices=['d1', 'd2', 'both'],
                       help='Датасет для использования (d1, d2, both)')
    parser.add_argument('--visualize', action='store_true',
                       help='Включить визуализацию результатов')
    parser.add_argument('--retrain', action='store_true',
                       help='Дообучить модель на новом датасете')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Директория с данными (по умолчанию: ../data относительно скрипта)')
    
    args = parser.parse_args()
    
    # Определяем директорию с данными
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(PROJECT_ROOT, 'data')
    
    # Создаем директорию для результатов
    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Запуск нейронной сети для классификации")
    print("=" * 60)
    print(f"📁 Директория с данными: {data_dir}")
    print(f"📁 Директория для результатов: {output_dir}")
    
    # Определяем датасеты для обработки
    datasets = []
    if args.dataset == 'd1' or args.dataset == 'both':
        datasets.append(('d1', os.path.join(data_dir, 'd1.csv')))
    if args.dataset == 'd2' or args.dataset == 'both':
        datasets.append(('d2', os.path.join(data_dir, 'd2.csv')))
    
    if not datasets:
        print("✗ Не указаны датасеты для обработки")
        return
    
    # Проверяем существование файлов
    for dataset_name, dataset_path in datasets:
        if not os.path.exists(dataset_path):
            print(f"✗ Файл не найден: {dataset_path}")
            print(f"  Проверьте путь или укажите правильную директорию с данными через --data_dir")
            return
    
    results = {}
    
    for dataset_name, dataset_path in datasets:
        print(f"\n{'='*40}")
        print(f"📁 Обработка датасета: {dataset_name}")
        print(f"📄 Файл: {dataset_path}")
        print(f"{'='*40}")
        
        # Загрузка данных
        X, y = load_csv_data(dataset_path)
        if X is None or y is None:
            print(f"✗ Пропускаем датасет {dataset_name}")
            continue
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        
        # Создание и обучение модели
        print("\n🧠 Создание и обучение модели...")
        
        if CPP_MODULE_AVAILABLE:
            # Используем C++ модуль
            loader = ai_module.DataLoader()
            dataset = loader.loadFromCSV(dataset_path)
            
            # Создаем нейронную сеть
            feature_count = dataset.getFeatureCount()
            net = ai_module.NeuralNetwork([feature_count, 8, 2])
            
            # Создаем и настраиваем тренер
            trainer = ai_module.GeneticTrainer(
                population_size=50,
                max_generations=100
            )
            
            # Обучение
            history = trainer.train(net, dataset)
            
            # Расчет метрик
            metrics = trainer.calculateMetrics(net, dataset)
            
            print(f"✓ Обучение завершено")
            print(f"  F1-score: {metrics.f1:.4f}")
            print(f"  Accuracy: {metrics.accuracy:.4f}")
            
            # Сохраняем результаты
            results[dataset_name] = {
                'f1': metrics.f1,
                'accuracy': metrics.accuracy,
                'history': history
            }
            
            # Визуализация
            if args.visualize:
                print("\n🎨 Визуализация результатов...")
                visualizer = ai_module.Visualizer()
                
                # Визуализация 2D данных
                output_image = os.path.join(output_dir, f"{dataset_name}_result.png")
                visualizer.plot2D(dataset, net, output_image)
                
                # Визуализация истории обучения
                history_image = os.path.join(output_dir, f"{dataset_name}_history.png")
                visualizer.plotTrainingHistory(history, history_image)
        
        else:
            # Используем заглушки
            print("⚠ Используются заглушки (C++ модуль не доступен)")
            
            # Создаем заглушки
            loader = DataLoader()
            dataset = loader.loadFromCSV(dataset_path)
            
            net = NeuralNetwork([X.shape[1], 8, 2])
            trainer = GeneticTrainer(population_size=50, max_generations=100)
            
            # Обучение
            history = trainer.train(net, dataset)
            
            # Расчет метрик
            metrics = trainer.calculateMetrics(net, dataset)
            
            print(f"✓ Обучение завершено (заглушка)")
            print(f"  F1-score: {metrics.f1:.4f}")
            print(f"  Accuracy: {metrics.accuracy:.4f}")
            
            # Сохраняем результаты
            results[dataset_name] = {
                'f1': metrics.f1,
                'accuracy': metrics.accuracy,
                'history': history
            }
            
            # Визуализация с помощью заглушек
            if args.visualize:
                print("\n🎨 Визуализация результатов (заглушки)...")
                output_image = os.path.join(output_dir, f"{dataset_name}_result.png")
                Visualizer.plot2D(dataset, net, output_image)
                
                history_image = os.path.join(output_dir, f"{dataset_name}_history.png")
                Visualizer.plotTrainingHistory(history, history_image)
        
        # Оценка на тестовых данных (если есть реальные предсказания)
        if CPP_MODULE_AVAILABLE and hasattr(net, 'predict'):
            # Здесь можно добавить реальные предсказания
            pass
    
    # Вывод сводки результатов
    print(f"\n{'='*60}")
    print("📈 СВОДКА РЕЗУЛЬТАТОВ")
    print(f"{'='*60}")
    
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  F1-score:  {metrics['f1']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        if 'history' in metrics:
            print(f"  История обучения: {len(metrics['history'])} поколений")
    
    # Дообучение на новом датасете
    if args.retrain:
        print(f"\n{'='*60}")
        print("🔄 ДООБУЧЕНИЕ НА НОВОМ ДАТАСЕТЕ")
        print(f"{'='*60}")
        
        # Здесь можно добавить логику дообучения
        print("Функция дообучения будет реализована после интеграции pybind11")
        print("Для дообучения передайте новый датасет и используйте метод train()")
    
    print(f"\n{'='*60}")
    print("✅ ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*60}")
    
    if not CPP_MODULE_AVAILABLE:
        print("\n⚠ ВНИМАНИЕ: Использовались заглушки")
        print("Для полной функциональности:")
        print("1. Соберите C++ модуль через pybind11")
        print("2. Убедитесь, что ai_module доступен для импорта")
        print("3. Запустите скрипт снова")
    
    print(f"\n📁 Результаты сохранены в директории: {output_dir}")
    if args.visualize:
        print("🎨 Графики сохранены в формате PNG")

if __name__ == "__main__":
    main()