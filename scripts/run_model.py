import sys
import os
import argparse
import subprocess
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Получаем абсолютный путь к директории скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Определяем платформу
IS_WINDOWS = platform.system() == "Windows"
EXE_EXTENSION = ".exe" if IS_WINDOWS else ""

def find_executable():
    """Находит исполняемый файл C++ программы"""
    # Определяем возможные пути к исполняемому файлу
    possible_paths = []
    
    # Основные пути для Windows (Visual Studio)
    if IS_WINDOWS:
        possible_paths.extend([
            os.path.join(PROJECT_ROOT, "build", "Debug", f"main{EXE_EXTENSION}"),
            os.path.join(PROJECT_ROOT, "build", "Release", f"main{EXE_EXTENSION}"),
            os.path.join(PROJECT_ROOT, "build", f"main{EXE_EXTENSION}"),
            os.path.join(PROJECT_ROOT, f"main{EXE_EXTENSION}"),
            os.path.join(PROJECT_ROOT, "Debug", f"main{EXE_EXTENSION}"),
            os.path.join(PROJECT_ROOT, "Release", f"main{EXE_EXTENSION}")
        ])
    else:
        # Пути для Linux/macOS
        possible_paths.extend([
            os.path.join(PROJECT_ROOT, "build", "main"),
            os.path.join(PROJECT_ROOT, "main")
        ])
    
    # Ищем существующий файл
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def check_build(exe_path=None):
    """Проверяет, собран ли C++ проект"""
    if exe_path:
        # Используем указанный пользователем путь
        if os.path.exists(exe_path):
            print(f"✓ Исполняемый файл найден: {exe_path}")
            return exe_path
        else:
            print(f"✗ Указанный файл не найден: {exe_path}")
            return None
    
    # Автоматический поиск
    exe_path = find_executable()
    
    if exe_path is None:
        print("✗ Исполняемый файл не найден. Проверенные пути:")
        
        # Показываем возможные пути
        if IS_WINDOWS:
            print("  Для Windows ожидаются пути:")
            print(f"    - {os.path.join(PROJECT_ROOT, 'build', 'Debug', 'main.exe')}")
            print(f"    - {os.path.join(PROJECT_ROOT, 'build', 'Release', 'main.exe')}")
            print(f"    - {os.path.join(PROJECT_ROOT, 'build', 'main.exe')}")
            print("\n  Если вы используете Visual Studio:")
            print("    - Соберите проект в конфигурации Debug или Release")
            print("    - Исполняемый файл будет в папке build/Debug/ или build/Release/")
        else:
            print("  Для Linux/macOS:")
            print(f"    - {os.path.join(PROJECT_ROOT, 'build', 'main')}")
            print("\n  Соберите проект: mkdir build && cd build && cmake .. && make")
        
        print("\n  Вы можете указать путь вручную через --exe_path")
        return None
    
    print(f"✓ Исполняемый файл найден: {exe_path}")
    return exe_path

def run_cpp_program(exe_path, algorithm="gradient|genetic"):
    """Запускает C++ программу и возвращает результаты"""
    print(f"\nЗапуск C++ программы с алгоритмом: {algorithm}")
    print(f"   Исполняемый файл: {exe_path}")
    
    try:
        # Запускаем процесс
        process = subprocess.Popen(
            [exe_path],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Читаем вывод в реальном времени
        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())
                output_lines.append(line.strip())
        
        # Получаем оставшийся вывод
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"✗ Ошибка выполнения C++ программы (код: {process.returncode})")
            if stderr:
                print(f"  Ошибка: {stderr}")
            return None
        
        print("✓ C++ программа успешно выполнена")
        return output_lines
        
    except Exception as e:
        print(f"✗ Ошибка запуска C++ программы: {e}")
        return None

def load_results_from_csv():
    """Загружает результаты из CSV файлов, созданных C++ программой"""
    results_dir = os.path.join(PROJECT_ROOT, "results")
    if not os.path.exists(results_dir):
        print(f"✗ Директория с результатами не найдена: {results_dir}")
        return {}
    
    results = {}
    
    # Ищем CSV файлы
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        dataset_name = csv_file.replace('.csv', '')
        file_path = os.path.join(results_dir, csv_file)
        
        try:
            df = pd.read_csv(file_path)
            results[dataset_name] = df
            print(f"✓ Загружен {csv_file}: {df.shape[0]} строк, {df.shape[1]} столбцов")
        except Exception as e:
            print(f"✗ Ошибка загрузки {csv_file}: {e}")
    
    return results

def visualize_2d_data(X, y, predictions=None, output_path="2d_visualization.png", title="2D Data Visualization"):
    """Визуализация 2D данных"""
    plt.figure(figsize=(10, 8))
    
    if predictions is not None:
        # Разделяем правильно и неправильно классифицированные точки
        correct_mask = (y == predictions)
        incorrect_mask = ~correct_mask
        
        # Правильно классифицированные
        plt.scatter(X[correct_mask, 0], X[correct_mask, 1], 
                   c=y[correct_mask], cmap=plt.cm.RdYlBu,
                   marker='o', s=100, alpha=0.7, label='Correct', edgecolors='k')
        
        # Неправильно классифицированные
        plt.scatter(X[incorrect_mask, 0], X[incorrect_mask, 1],
                   c='black', marker='x', s=150, alpha=1.0, label='Incorrect', linewidths=2)
        
        accuracy = np.mean(correct_mask)
        plt.title(f"{title}\nAccuracy: {accuracy:.2%}")
        plt.legend()
    else:
        # Просто точки данных
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                             edgecolors='k', s=100, alpha=0.7)
        plt.title(title)
        plt.colorbar(scatter, label='Class')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Визуализация сохранена: {output_path}")

def visualize_multidimensional(X, y, output_path="multidimensional_visualization.png", title="Multidimensional Visualization"):
    """Визуализация многомерных данных с помощью PCA и t-SNE"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.RdYlBu,
                                  edgecolors='k', s=100, alpha=0.7)
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title(f'PCA (explained variance: {pca.explained_variance_ratio_.sum():.2%})')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Class')
    else:
        scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                                  edgecolors='k', s=100, alpha=0.7)
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].set_title('Original 2D Data')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Class')
    
    # t-SNE (только если больше 2 признаков и не слишком много данных)
    if X.shape[1] > 2 and len(X) <= 1000:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_tsne = tsne.fit_transform(X)
        scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.RdYlBu,
                                  edgecolors='k', s=100, alpha=0.7)
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].set_title('t-SNE Visualization')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Class')
    else:
        if X.shape[1] > 2:
            axes[1].text(0.5, 0.5, f"t-SNE skipped\n(too many samples: {len(X)})",
                        ha='center', va='center', fontsize=12)
        else:
            axes[1].text(0.5, 0.5, "t-SNE not needed\n(data is already 2D)",
                        ha='center', va='center', fontsize=12)
        axes[1].set_title('t-SNE')
        axes[1].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Многомерная визуализация сохранена: {output_path}")

def visualize_training_history(history, output_path="training_history.png", title="Training History"):
    """Визуализация истории обучения"""
    plt.figure(figsize=(10, 6))
    
    if isinstance(history, dict):
        # История из нескольки метрик
        for metric_name, values in history.items():
            plt.plot(values, label=metric_name, linewidth=2)
        plt.legend()
    else:
        # Простая история ошибок
        plt.plot(history, 'b-', linewidth=2, label='Loss')
        plt.legend()
    
    plt.xlabel('Generation/Epoch')
    plt.ylabel('Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  История обучения сохранена: {output_path}")

def evaluate_metrics(y_true, y_pred, dataset_name=""):
    """Оценка модели с помощью различных метрик"""
    if len(y_true) == 0 or len(y_pred) == 0:
        print("✗ Нет данных для оценки")
        return None
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    if dataset_name:
        print(f"\nРезультаты на {dataset_name}:")
    else:
        print(f"\nРезультаты оценки:")
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:\n{cm}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }

def load_dataset(dataset_path):
    """Загрузка датасета из CSV файла"""
    try:
        df = pd.read_csv(dataset_path)
        print(f"✓ Загружен датасет: {dataset_path}")
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
        return X, y, df
        
    except Exception as e:
        print(f"✗ Ошибка загрузки {dataset_path}: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser(description='Запуск нейронной сети для классификации')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['train', 'd1', 'd2', 'test', 'all'],
                       help='Датасет для использования (train, d1, d2, test, all)')
    parser.add_argument('--algorithm', type=str, default='gradient',
                       choices=['genetic', 'gradient'],
                       help='Алгоритм обучения (genetic, gradient)')
    parser.add_argument('--visualize', action='store_true',
                       help='Включить визуализацию результатов')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Директория с данными (по умолчанию: ../data относительно скрипта)')
    parser.add_argument('--skip_cpp', action='store_true',
                       help='Пропустить запуск C++ программы (использовать существующие результаты)')
    parser.add_argument('--exe_path', type=str, default=None,
                       help='Путь к исполняемому файлу (например: .\\build\\Debug\\main.exe)')
    
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
    print("Запуск нейронной сети для классификации")
    print("=" * 60)
    print(f"Директория с данными: {data_dir}")
    print(f"Директория для результатов: {output_dir}")
    print(f"Алгоритм обучения: {args.algorithm}")
    print(f"Визуализация: {'Включена' if args.visualize else 'Выключена'}")

    # Проверяем сборку C++ проекта
    exe_path = check_build(args.exe_path)
    if not args.skip_cpp and exe_path is None:
        print("✗ Не удалось найти исполняемый файл C++ проекта")
        print(f"  Укажите путь через --exe_path, например:")
        print(f"  --exe_path .\\build\\Debug\\main.exe")
        return
    
    # Запускаем C++ программу
    if not args.skip_cpp:
        output = run_cpp_program(exe_path, args.algorithm)
        if output is None:
            print("✗ Не удалось запустить C++ программу")
            return
    
    # Определяем датасеты для обработки
    datasets = []
    if args.dataset == 'train' or args.dataset == 'all':
        datasets.append(('train', os.path.join(data_dir, 'train.csv')))
    if args.dataset == 'd1' or args.dataset == 'all':
        datasets.append(('d1', os.path.join(data_dir, 'd1.csv')))
    if args.dataset == 'd2' or args.dataset == 'all':
        datasets.append(('d2', os.path.join(data_dir, 'd2.csv')))
    if args.dataset == 'test' or args.dataset == 'all':
        datasets.append(('test', os.path.join(data_dir, 'test.csv')))
    
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
        print(f"Обработка датасета: {dataset_name}")
        print(f"Файл: {dataset_path}")
        print(f"{'='*40}")
        
        # Загрузка данных
        X, y, df = load_dataset(dataset_path)
        if X is None:
            print(f"✗ Пропускаем датасет {dataset_name}")
            continue
        
        # Для test.csv нет меток, только предсказания
        if dataset_name == 'test':
            print(f"\nПредсказания для test.csv (без меток)")
            # Здесь можно добавить логику загрузки предсказаний из файлов
            continue
        
        # Визуализация
        if args.visualize:
            print(f"\nВизуализация датасета {dataset_name}...")
            
            if X.shape[1] == 2:
                # 2D визуализация
                output_image = os.path.join(output_dir, f"{dataset_name}_2d_data.png")
                visualize_2d_data(X, y, output_path=output_image, 
                                 title=f"{dataset_name} Dataset (2D)")
            else:
                # Многомерная визуализация
                output_image = os.path.join(output_dir, f"{dataset_name}_multidimensional.png")
                visualize_multidimensional(X, y, output_path=output_image,
                                         title=f"{dataset_name} Dataset ({X.shape[1]}D)")
        
        # Сохраняем информацию о датасете
        results[dataset_name] = {
            'X': X,
            'y': y,
            'df': df,
            'feature_count': X.shape[1],
            'sample_count': X.shape[0]
        }
    
    # Загружаем результаты из CSV файлов (если C++ программа их создала)
    csv_results = load_results_from_csv()
    
    # Объединяем результаты
    for dataset_name, csv_data in csv_results.items():
        if dataset_name in results:
            results[dataset_name]['csv_data'] = csv_data
    
    # Вывод сводки результатов
    print(f"\n{'='*60}")
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print(f"{'='*60}")
    
    for dataset_name, data in results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Примеров: {data['sample_count']}")
        print(f"  Признаков: {data['feature_count']}")
        
        if 'csv_data' in data:
            csv_shape = data['csv_data'].shape
            print(f"  Результатов из CSV: {csv_shape[0]} строк, {csv_shape[1]} столбцов")
    
    # Расчет итоговой оценки для d1 и d2 (если есть результаты)
    if 'd1' in results and 'd2' in results:
        print(f"\n{'='*40}")
        print("ИТОГОВАЯ ОЦЕНКА")
        print(f"{'='*40}")
        
        # Здесь можно добавить расчет итоговой оценки на основе реальных результатов
        print("  Для расчета итоговой оценки запустите C++ программу")
        print("  и используйте результаты из output файлов")
    
    print(f"\n{'='*60}")
    print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*60}")
    
    print(f"\nРезультаты сохранены в директории: {output_dir}")
    if args.visualize:
        print("Графики сохранены в формате PNG")


if __name__ == "__main__":
    main()