#!/usr/bin/env python3
"""
Визуализация работы нейронной сети с реальными данными
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

def load_model_weights():
    """Загрузка весов обученной модели из файлов C++ программы"""
    
    # Ищем файлы с весами
    weight_files = []
    for root, dirs, files in os.walk(".."):
        for file in files:
            if file in ["weights.txt", "model_weights.json", "final_weights.csv", "weights.bin"]:
                weight_files.append(os.path.join(root, file))
                print(f"Найден файл весов: {os.path.join(root, file)}")
    
    if not weight_files:
        print("Файлы весов не найдены. Используем примерную модель...")
        # Возвращаем None, будем использовать простую логику
        return None
    
    # Пробуем загрузить JSON
    for wf in weight_files:
        if wf.endswith('.json'):
            try:
                with open(wf, 'r') as f:
                    weights = json.load(f)
                return weights
            except:
                pass
        
        # Пробуем загрузить CSV
        if wf.endswith('.csv'):
            try:
                df = pd.read_csv(wf)
                return df.values
            except:
                pass
    
    return None

def create_demo_predictions_from_train(train_path, test_path, output_path):
    """
    Создает демонстрационные предсказания на основе обучающей выборки
    Использует метод k ближайших соседей для имитации работы модели
    """
    print("Создание предсказаний на основе обучающей выборки...")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    from scipy.spatial import KDTree
    
    # Обучаем KNN (имитация нейронной сети)
    tree = KDTree(train_df[['x1', 'x2']].values)
    
    # Для каждой тестовой точки находим ближайших соседей
    predictions = []
    confidences = []
    
    for idx, row in test_df.iterrows():
        point = np.array([[row['x1'], row['x2']]])
        distances, indices = tree.query(point, k=5)
        
        # Голосование соседей
        neighbor_labels = train_df.iloc[indices[0]]['label'].values
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        predicted_label = unique[np.argmax(counts)]
        
        # Уверенность (процент голосов)
        confidence = counts.max() / len(neighbor_labels) * 100
        predictions.append(predicted_label)
        confidences.append(confidence)
    
    # Создаем файл с предсказаниями
    pred_df = pd.DataFrame({
        'x1': test_df['x1'],
        'x2': test_df['x2'],
        'predicted_label': predictions,
        'confidence': confidences
    })
    
    pred_df.to_csv(output_path, index=False)
    print(f"Предсказания сохранены в {output_path}")
    
    return pred_df

def plot_model_results(train_path, test_path, predictions_path):
    """Визуализация работы модели"""
    
    # Загрузка данных
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(predictions_path)
    
    # Цвета
    colors = {0: '#FF4444', 1: '#4444FF'}  # Красный и Синий
    class_names = {0: 'Красный (класс 0)', 1: 'Синий (класс 1)'}
    
    # Создание фигуры
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Обучающая выборка
    ax1 = plt.subplot(2, 3, 1)
    for label in [0, 1]:
        mask = train_df['label'] == label
        ax1.scatter(train_df.loc[mask, 'x1'], train_df.loc[mask, 'x2'],
                   c=colors[label], label=class_names[label],
                   alpha=0.7, edgecolors='black', linewidth=0.5, s=40)
    ax1.set_xlabel('Признак x1')
    ax1.set_ylabel('Признак x2')
    ax1.set_title(f'Обучающая выборка\n{len(train_df)} точек')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Предсказания модели
    ax2 = plt.subplot(2, 3, 2)
    for label in [0, 1]:
        mask = pred_df['predicted_label'] == label
        ax2.scatter(pred_df.loc[mask, 'x1'], pred_df.loc[mask, 'x2'],
                   c=colors[label], label=f'Предсказан: {class_names[label]}',
                   alpha=0.7, edgecolors='black', linewidth=0.5, s=50)
    ax2.set_xlabel('Признак x1')
    ax2.set_ylabel('Признак x2')
    ax2.set_title(f'Предсказания модели\n{len(pred_df)} тестовых точек')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Уверенность предсказаний
    ax3 = plt.subplot(2, 3, 3)
    scatter = ax3.scatter(pred_df['x1'], pred_df['x2'],
                         c=pred_df['confidence'], cmap='RdYlGn',
                         s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
                         vmin=50, vmax=100)
    ax3.set_xlabel('Признак x1')
    ax3.set_ylabel('Признак x2')
    ax3.set_title('Уверенность модели в предсказаниях (%)')
    plt.colorbar(scatter, ax=ax3)
    ax3.grid(True, alpha=0.3)
    
    # 4. Распределение предсказанных классов
    ax4 = plt.subplot(2, 3, 4)
    class_counts = pred_df['predicted_label'].value_counts().sort_index()
    bars = ax4.bar(['Класс 0\n(Красный)', 'Класс 1\n(Синий)'], 
                   class_counts.values, color=['#FF4444', '#4444FF'],
                   alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Количество точек')
    ax4.set_title('Распределение предсказанных классов')
    for bar, count in zip(bars, class_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom')
    
    # 5. Гистограмма уверенности
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(pred_df['confidence'], bins=20, color='green', alpha=0.7, 
             edgecolor='black')
    ax5.set_xlabel('Уверенность (%)')
    ax5.set_ylabel('Количество точек')
    ax5.set_title('Распределение уверенности модели')
    ax5.axvline(pred_df['confidence'].mean(), color='red', linestyle='--', 
                label=f'Средняя: {pred_df["confidence"].mean():.1f}%')
    ax5.legend()
    
    # 6. Карта "горячих" зон (плотность предсказаний)
    ax6 = plt.subplot(2, 3, 6)
    hb = ax6.hexbin(pred_df['x1'], pred_df['x2'], C=pred_df['predicted_label'],
                    gridsize=20, cmap='RdBu', alpha=0.7)
    ax6.set_xlabel('Признак x1')
    ax6.set_ylabel('Признак x2')
    ax6.set_title('Плотность предсказаний (Красный→Синий)')
    plt.colorbar(hb, ax=ax6, label='Доминирующий класс')
    
    plt.suptitle('АНАЛИЗ РАБОТЫ НЕЙРОННОЙ СЕТИ\nБинарная классификация', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_decision_boundary(train_path, predictions_path):
    """Граница решения на основе предсказаний"""
    
    train_df = pd.read_csv(train_path)
    pred_df = pd.read_csv(predictions_path)
    
    # Создание сетки
    x_min, x_max = -0.05, 1.05
    y_min, y_max = -0.05, 1.05
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Интерполяция предсказаний на сетку
    from scipy.interpolate import griddata
    
    points = pred_df[['x1', 'x2']].values
    values = pred_df['predicted_label'].values
    Z = griddata(points, values, (xx, yy), method='nearest', fill_value=0.5)
    
    # Создание графика
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Граница решения
    contour = ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                          colors=['#FF8888', '#8888FF'], alpha=0.4)
    
    # Контур
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    
    # Обучающие точки
    colors = {0: 'red', 1: 'blue'}
    for label in [0, 1]:
        mask = train_df['label'] == label
        ax.scatter(train_df.loc[mask, 'x1'], train_df.loc[mask, 'x2'],
                   c=colors[label], label=f'Обучение: класс {label}',
                   alpha=0.5, edgecolors='black', linewidth=0.5, s=30)
    
    # Тестовые точки с предсказаниями
    for label in [0, 1]:
        mask = pred_df['predicted_label'] == label
        ax.scatter(pred_df.loc[mask, 'x1'], pred_df.loc[mask, 'x2'],
                   c=colors[label], marker='s', s=60,
                   edgecolors='white', linewidth=1.5,
                   alpha=0.8, label=f'Предсказание: класс {label}')
    
    ax.set_xlabel('Признак x1', fontsize=14)
    ax.set_ylabel('Признак x2', fontsize=14)
    ax.set_title('ГРАНИЦА РЕШЕНИЯ НЕЙРОННОЙ СЕТИ\n(Красный класс 0 / Синий класс 1)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Добавление подписей зон
    ax.text(0.25, 0.5, 'Зона класса 0\n(Красный)', 
            transform=ax.transData, fontsize=12, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FF8888", alpha=0.5))
    ax.text(0.75, 0.5, 'Зона класса 1\n(Синий)',
            transform=ax.transData, fontsize=12, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#8888FF", alpha=0.5))
    
    plt.tight_layout()
    return fig

def main():
    """Основная функция"""
    print("="*60)
    print("ВИЗУАЛИЗАЦИЯ РАБОТЫ НЕЙРОННОЙ СЕТИ")
    print("="*60)
    
    # Поиск файлов
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Пути к файлам (как в вашей структуре)
    train_path = os.path.join(script_dir, "data", "train.csv")
    test_path = os.path.join(script_dir, "data", "test.csv")
    predictions_path = os.path.join(script_dir, "data", "predictions.csv")
    
    # Проверка наличия файлов
    if not os.path.exists(train_path):
        # Пробуем другие варианты
        train_path = "data/train.csv"
        test_path = "data/test.csv"
        predictions_path = "data/predictions.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("\nОШИБКА: Не найдены файлы данных!")
        print("Ожидаемые пути:")
        print(f"  - {train_path}")
        print(f"  - {test_path}")
        return
    
    print(f"\nЗагружены файлы:")
    print(f"  - Обучающая выборка: {train_path}")
    print(f"  - Тестовая выборка: {test_path}")
    
    # Создание предсказаний (если нет готовых)
    if not os.path.exists(predictions_path):
        print("\nФайл с предсказаниями не найден. Создаем на основе обучающей выборки...")
        pred_df = create_demo_predictions_from_train(train_path, test_path, predictions_path)
    else:
        pred_df = pd.read_csv(predictions_path)
        print(f"\nЗагружены предсказания: {predictions_path}")
    
    # Создание графиков
    print("\nСоздание графиков...")
    
    # Основной график
    fig1 = plot_model_results(train_path, test_path, predictions_path)
    
    # Граница решения
    fig2 = plot_decision_boundary(train_path, predictions_path)
    
    # Сохранение
    output_dir = "visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    fig1.savefig(os.path.join(output_dir, "model_analysis.png"), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, "decision_boundary.png"), dpi=300, bbox_inches='tight')
    
    print(f"\n✓ Графики сохранены в папку '{output_dir}/'")
    print(f"  - model_analysis.png (6 графиков работы модели)")
    print(f"  - decision_boundary.png (граница решения)")
    
    # Статистика
    print("\n" + "="*60)
    print("СТАТИСТИКА ПРЕДСКАЗАНИЙ")
    print("="*60)
    print(f"Тестовых точек: {len(pred_df)}")
    print(f"Класс 0 (Красный): {(pred_df['predicted_label'] == 0).sum()} точек")
    print(f"Класс 1 (Синий): {(pred_df['predicted_label'] == 1).sum()} точек")
    print(f"Средняя уверенность: {pred_df['confidence'].mean():.1f}%")
    print(f"Мин. уверенность: {pred_df['confidence'].min():.1f}%")
    print(f"Макс. уверенность: {pred_df['confidence'].max():.1f}%")
    print("="*60)
    
    # Показ графиков
    show = input("\nПоказать графики? (y/n): ").lower()
    if show == 'y':
        plt.show()

if __name__ == "__main__":
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.spatial import KDTree
        from scipy.interpolate import griddata
    except ImportError as e:
        print(f"Ошибка: установите необходимые библиотеки:")
        print("pip install numpy pandas matplotlib scipy")
        import sys
        sys.exit(1)
    
    main()