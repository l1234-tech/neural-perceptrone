#!/usr/bin/env python3
"""
Простой скрипт для визуализации результатов тестирования нейронной сети.
Запуск: python plot_results.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from pathlib import Path

def setup_plot_style():
    """Настройка стиля графиков"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16

def plot_dataset(data_file, title="Датасет"):
    """Визуализация датасета"""
    if not os.path.exists(data_file):
        print(f"Файл {data_file} не найден")
        return None
    
    try:
        data = pd.read_csv(data_file)
        # Проверяем разные возможные названия столбцов
        if 'x1' in data.columns and 'x2' in data.columns:
            x_col, y_col = 'x1', 'x2'
        elif 'feature_0' in data.columns and 'feature_1' in data.columns:
            x_col, y_col = 'feature_0', 'feature_1'
        else:
            print(f"Файл {data_file} не содержит нужных столбцов")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if 'label' in data.columns:
            # Цветовая кодировка по классам
            scatter = ax.scatter(data[x_col], data[y_col], 
                               c=data['label'], cmap='coolwarm', 
                               edgecolors='k', s=50, alpha=0.7)
            plt.colorbar(scatter, label='Класс')
            ax.set_title(f'{title} (размер: {len(data)} точек)')
        else:
            # Только точки без меток
            ax.scatter(data[x_col], data[y_col], 
                      c='blue', edgecolors='k', s=50, alpha=0.7)
            ax.set_title(f'{title} (без меток, размер: {len(data)} точек)')
        
        ax.set_xlabel('Признак 0 (x1)')
        ax.set_ylabel('Признак 1 (x2)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Ошибка при чтении файла {data_file}: {e}")
        return None

def plot_predictions(predictions_file, title="Предсказания модели"):
    """Визуализация предсказаний модели"""
    if not os.path.exists(predictions_file):
        print(f"Файл {predictions_file} не найден")
        return None
    
    try:
        data = pd.read_csv(predictions_file)
        
        # Определяем названия столбцов
        if 'x1' in data.columns and 'x2' in data.columns:
            x_col, y_col = 'x1', 'x2'
        elif 'feature_0' in data.columns and 'feature_1' in data.columns:
            x_col, y_col = 'feature_0', 'feature_1'
        else:
            print(f"Файл {predictions_file} не содержит нужных столбцов")
            return None
        
        required_cols = ['true_label', 'predicted_label']
        if not all(col in data.columns for col in required_cols):
            print(f"Файл {predictions_file} не содержит всех нужных столбцов")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Истинные метки
        scatter1 = axes[0].scatter(data[x_col], data[y_col], 
                                  c=data['true_label'], cmap='coolwarm',
                                  edgecolors='k', s=50, alpha=0.7)
        axes[0].set_xlabel('Признак 0 (x1)')
        axes[0].set_ylabel('Признак 1 (x2)')
        axes[0].set_title('Истинные метки')
        axes[0].grid(True, alpha=0.3)
        
        # Предсказанные метки
        scatter2 = axes[1].scatter(data[x_col], data[y_col], 
                                  c=data['predicted_label'], cmap='coolwarm',
                                  edgecolors='k', s=50, alpha=0.7)
        axes[1].set_xlabel('Признак 0 (x1)')
        axes[1].set_ylabel('Признак 1 (x2)')
        axes[1].set_title('Предсказанные метки')
        axes[1].grid(True, alpha=0.3)
        
        # Добавляем цветовые шкалы
        plt.colorbar(scatter1, ax=axes[0], label='Класс')
        plt.colorbar(scatter2, ax=axes[1], label='Класс')
        
        # Вычисляем точность
        correct = (data['true_label'] == data['predicted_label']).sum()
        accuracy = correct / len(data) * 100
        
        fig.suptitle(f'{title} (Точность: {accuracy:.2f}%)', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        print(f"Ошибка при чтении файла {predictions_file}: {e}")
        return None

def plot_training_history(history_file, title="История обучения"):
    """Визуализация истории обучения"""
    if not os.path.exists(history_file):
        print(f"Файл {history_file} не найден")
        return None
    
    try:
        data = pd.read_csv(history_file)
        if 'generation' not in data.columns or 'loss' not in data.columns:
            print(f"Файл {history_file} не содержит нужных столбцов")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(data['generation'], data['loss'], linewidth=2, color='red')
        ax.set_xlabel('Поколение')
        ax.set_ylabel('Ошибка (loss)')
        ax.set_title(f'{title} ({len(data)} поколений)')
        ax.grid(True, alpha=0.3)
        
        # Добавляем аннотацию с улучшением
        if len(data) > 1:
            initial_loss = data['loss'].iloc[0]
            final_loss = data['loss'].iloc[-1]
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            ax.annotate(f'Начальная ошибка: {initial_loss:.4f}\n'
                       f'Финальная ошибка: {final_loss:.4f}\n'
                       f'Улучшение: {improvement:.1f}%',
                       xy=(0.7, 0.9), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Ошибка при чтении файла {history_file}: {e}")
        return None

def plot_decision_boundary(boundary_file, title="Граница решения"):
    """Визуализация границы решения"""
    if not os.path.exists(boundary_file):
        print(f"Файл {boundary_file} не найден")
        return None
    
    try:
        data = pd.read_csv(boundary_file)
        
        # Определяем названия столбцов
        if 'x1' in data.columns and 'x2' in data.columns:
            x_col, y_col = 'x1', 'x2'
        elif 'x' in data.columns and 'y' in data.columns:
            x_col, y_col = 'x', 'y'
        else:
            print(f"Файл {boundary_file} не содержит нужных столбцов")
            return None
        
        if 'prediction' not in data.columns:
            print(f"Файл {boundary_file} не содержит столбца prediction")
            return None
        
        # Определяем размер сетки
        x_unique = data[x_col].unique()
        grid_size = len(x_unique)
        
        # Преобразуем в матрицу для contourf
        x_grid = data[x_col].values.reshape(grid_size, grid_size)
        y_grid = data[y_col].values.reshape(grid_size, grid_size)
        z_grid = data['prediction'].values.reshape(grid_size, grid_size)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Граница решения
        contour = ax.contourf(x_grid, y_grid, z_grid, levels=[-0.5, 0.5, 1.5], 
                             alpha=0.3, cmap='coolwarm')
        
        # Контуры
        ax.contour(x_grid, y_grid, z_grid, levels=[0.5], colors='black', linewidths=2)
        
        ax.set_xlabel('Признак 0 (x1)')
        ax.set_ylabel('Признак 1 (x2)')
        ax.set_title(f'{title} ({grid_size}x{grid_size} точек)')
        ax.grid(True, alpha=0.3)
        
        # Добавляем цветовую шкалу
        plt.colorbar(contour, ax=ax, label='Предсказанный класс')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Ошибка при чтении файла {boundary_file}: {e}")
        return None

def plot_confusion_matrix_from_predictions(predictions_file, title="Матрица ошибок"):
    """Визуализация матрицы ошибок из файла предсказаний"""
    if not os.path.exists(predictions_file):
        print(f"Файл {predictions_file} не найден")
        return None
    
    try:
        data = pd.read_csv(predictions_file)
        if 'true_label' not in data.columns or 'predicted_label' not in data.columns:
            print(f"Файл {predictions_file} не содержит нужных столбцов")
            return None
        
        # Вычисляем матрицу ошибок
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(data['true_label'], data['predicted_label'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Тепловая карта
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Предсказано 0', 'Предсказано 1'],
                   yticklabels=['Истинный 0', 'Истинный 1'],
                   ax=ax)
        
        ax.set_xlabel('Предсказанный класс')
        ax.set_ylabel('Истинный класс')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        print(f"Ошибка при чтении файла {predictions_file}: {e}")
        return None

def display_report(report_file):
    """Отображение текстового отчета"""
    if not os.path.exists(report_file):
        print(f"Файл отчета {report_file} не найден")
        return
    
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            report_text = f.read()
        
        print("\n" + "="*60)
        print("ОТЧЕТ О ТЕСТИРОВАНИИ")
        print("="*60)
        print(report_text)
        
    except Exception as e:
        print(f"Ошибка при чтении файла отчета {report_file}: {e}")

def main():
    """Основная функция"""
    print("Визуализация результатов тестирования нейронной сети")
    print("="*60)
    
    # Настройка стиля графиков
    setup_plot_style()
    
    # Определяем директорию, где находится скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Список доступных файлов в директории скрипта
    available_files = []
    for file in os.listdir(script_dir):
        if file.endswith('.csv') or file.endswith('.txt'):
            available_files.append(file)
    
    if not available_files:
        print(f"В папке {script_dir} не найдено файлов для визуализации")
        print("Запустите сначала программу на C++ для генерации данных")
        print("\nОжидаемые файлы:")
        print("  - train_data.csv")
        print("  - test_data.csv")
        print("  - test_predictions.csv")
        print("  - training_history.csv")
        print("  - decision_boundary.csv")
        print("  - test_report.txt")
        return
    
    print(f"Найдены файлы в {script_dir}:")
    for i, file in enumerate(available_files, 1):
        print(f"  {i}. {file}")
    
    # Автоматическое определение типов файлов
    figures = []
    
    # 1. Обучающие данные
    train_files = [f for f in available_files if 'train' in f.lower() and f.endswith('.csv')]
    for train_file in train_files:
        filepath = os.path.join(script_dir, train_file)
        fig = plot_dataset(filepath, "Обучающая выборка")
        if fig:
            figures.append((fig, f"train_{train_file}"))
    
    # 2. Тестовые данные
    test_files = [f for f in available_files if 'test' in f.lower() and f.endswith('.csv') and 'prediction' not in f.lower()]
    for test_file in test_files:
        filepath = os.path.join(script_dir, test_file)
        fig = plot_dataset(filepath, "Тестовая выборка")
        if fig:
            figures.append((fig, f"test_{test_file}"))
    
    # 3. Предсказания
    pred_files = [f for f in available_files if 'prediction' in f.lower() and f.endswith('.csv')]
    for pred_file in pred_files:
        filepath = os.path.join(script_dir, pred_file)
        fig = plot_predictions(filepath, "Предсказания модели")
        if fig:
            figures.append((fig, f"predictions_{pred_file}"))
        
        # Матрица ошибок из предсказаний
        cm_fig = plot_confusion_matrix_from_predictions(filepath, "Матрица ошибок")
        if cm_fig:
            figures.append((cm_fig, f"confusion_matrix_{pred_file}"))
    
    # 4. История обучения
    history_files = [f for f in available_files if 'history' in f.lower() and f.endswith('.csv')]
    for history_file in history_files:
        filepath = os.path.join(script_dir, history_file)
        fig = plot_training_history(filepath, "История обучения")
        if fig:
            figures.append((fig, f"history_{history_file}"))
    
    # 5. Граница решения
    boundary_files = [f for f in available_files if 'boundary' in f.lower() and f.endswith('.csv')]
    for boundary_file in boundary_files:
        filepath = os.path.join(script_dir, boundary_file)
        fig = plot_decision_boundary(filepath, "Граница решения")
        if fig:
            figures.append((fig, f"boundary_{boundary_file}"))
    
    # 6. Отчеты
    report_files = [f for f in available_files if f.endswith('.txt')]
    for report_file in report_files:
        filepath = os.path.join(script_dir, report_file)
        display_report(filepath)
    
    # Сохранение графиков
    if figures:
        print(f"\nСоздано {len(figures)} графиков")
        
        # Создаем папку для сохранения
        output_dir = os.path.join(script_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (fig, name) in enumerate(figures, 1):
            filename = os.path.join(output_dir, f"{name}.png")
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  {i}. График сохранен как {filename}")
        
        print(f"\nВсе графики сохранены в папке '{output_dir}/'")
        
        # Показываем графики
        show = input("\nПоказать графики на экране? (y/n): ").lower()
        if show == 'y':
            plt.show()
    else:
        print("\nНе удалось создать графики. Проверьте наличие корректных данных.")
    
    print("\nЗавершено!")

if __name__ == "__main__":
    # Проверяем зависимости
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Ошибка: отсутствуют необходимые библиотеки Python")
        print(f"Установите их командой: pip install numpy pandas matplotlib scikit-learn seaborn")
        sys.exit(1)
    
    main()
