#include "visualization/Visualizer.h"
#include "data/DataPoint.h"
#include "utils/Vector.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>

using namespace std;

// Вспомогательная функция для проверки существования файла
static bool fileExists(const std::string& filename) {
    ifstream file(filename);
    return file.good();
}

// Генерация сетки точек для построения границы решения
vector<pair<double, double>> Visualizer::generateGrid(
    const DataSet& data, 
    int gridSize,
    double margin) {
    
    if (data.empty()) {
        return {};
    }
    
    // Получаем первую точку для определения размерности
    const DataPoint& firstPoint = data.get(0);
    const vector<double>& firstFeatures = firstPoint.getFeatures();
    
    if (firstFeatures.size() != 2) {
        cerr << "Visualizer::generateGrid: Data must be 2D for visualization" << endl;
        return {};
    }
    
    // Находим минимальные и максимальные значения по осям X и Y
    double minX = firstFeatures[0];
    double maxX = firstFeatures[0];
    double minY = firstFeatures[1];
    double maxY = firstFeatures[1];
    
    for (size_t i = 1; i < data.size(); ++i) {
        const DataPoint& point = data.get(i);
        const vector<double>& features = point.getFeatures();
        
        if (features.size() != 2) {
            cerr << "Visualizer::generateGrid: Point " << i << " is not 2D" << endl;
            continue;
        }
        
        minX = min(minX, features[0]);
        maxX = max(maxX, features[0]);
        minY = min(minY, features[1]);
        maxY = max(maxY, features[1]);
    }
    
    // Добавляем отступы
    double rangeX = maxX - minX;
    double rangeY = maxY - minY;
    minX -= rangeX * margin;
    maxX += rangeX * margin;
    minY -= rangeY * margin;
    maxY += rangeY * margin;
    
    // Генерируем сетку
    vector<pair<double, double>> grid;
    grid.reserve(gridSize * gridSize);
    
    double stepX = (maxX - minX) / (gridSize - 1);
    double stepY = (maxY - minY) / (gridSize - 1);
    
    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            double x = minX + i * stepX;
            double y = minY + j * stepY;
            grid.emplace_back(x, y);
        }
    }
    
    return grid;
}

// Сохранение данных точек в CSV файл
void Visualizer::savePointsToCSV(const DataSet& data, 
                               const std::string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Visualizer::savePointsToCSV: Cannot open file " << filename << endl;
        return;
    }
    
    // Заголовок CSV
    file << "x,y,label,prediction" << endl;
    
    for (size_t i = 0; i < data.size(); ++i) {
        const DataPoint& point = data.get(i);
        const vector<double>& features = point.getFeatures();
        
        if (features.size() != 2) {
            cerr << "Visualizer::savePointsToCSV: Point " << i << " is not 2D, skipping" << endl;
            continue;
        }
        
        // Сохраняем координаты и метку
        file << features[0] << "," << features[1] << "," 
             << point.getLabel() << ",0" << endl; // prediction будет заполнен позже
    }
    
    file.close();
}

// Сохранение границы решения в CSV файл
void Visualizer::saveDecisionBoundaryToCSV(
    const NeuralNetwork& net,
    const vector<pair<double, double>>& grid,
    const std::string& filename) {
    
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Visualizer::saveDecisionBoundaryToCSV: Cannot open file " << filename << endl;
        return;
    }
    
    // Заголовок CSV
    file << "x,y,prediction" << endl;
    
    for (const auto& point : grid) {
        // Создаем вектор входных данных
        Vector input(2);
        input[0] = point.first;
        input[1] = point.second;
        
        // Получаем предсказание сети
        int prediction = net.predict(input);
        
        // Сохраняем координаты и предсказание
        file << point.first << "," << point.second << "," << prediction << endl;
    }
    
    file.close();
}

// Генерация Python скрипта для визуализации
void Visualizer::generateVisualizationScript(
    const std::string& pointsFile,
    const std::string& boundaryFile,
    const std::string& outputImage) {
    
    ofstream script("visualize_2d.py");
    if (!script.is_open()) {
        cerr << "Visualizer::generateVisualizationScript: Cannot create Python script" << endl;
        return;
    }
    
    script << "#!/usr/bin/env python3\n";
    script << "# Автоматически сгенерированный скрипт для визуализации 2D данных\n";
    script << "\n";
    script << "import pandas as pd\n";
    script << "import matplotlib.pyplot as plt\n";
    script << "import numpy as np\n";
    script << "import sys\n";
    script << "import os\n";
    script << "\n";
    script << "def main():\n";
    script << "    # Загрузка данных\n";
    script << "    try:\n";
    script << "        points_df = pd.read_csv('" << pointsFile << "')\n";
    script << "        boundary_df = pd.read_csv('" << boundaryFile << "')\n";
    script << "    except FileNotFoundError as e:\n";
    script << "        print(f'Ошибка: Файл не найден: {e}')\n";
    script << "        sys.exit(1)\n";
    script << "\n";
    script << "    # Создание графика\n";
    script << "    fig, ax = plt.subplots(figsize=(10, 8))\n";
    script << "\n";
    script << "    # Визуализация границы решения\n";
    script << "    if not boundary_df.empty:\n";
    script << "        # Преобразуем данные границы в сетку\n";
    script << "        x_unique = np.sort(boundary_df['x'].unique())\n";
    script << "        y_unique = np.sort(boundary_df['y'].unique())\n";
    script << "        X, Y = np.meshgrid(x_unique, y_unique)\n";
    script << "        \n";
    script << "        # Создаем массив предсказаний\n";
    script << "        Z = boundary_df.pivot(index='y', columns='x', values='prediction').values\n";
    script << "        \n";
    script << "        # Рисуем контур границы решения\n";
    script << "        contour = ax.contourf(X, Y, Z, alpha=0.3, cmap='coolwarm', levels=1)\n";
    script << "        ax.contour(X, Y, Z, colors='black', linewidths=0.5, levels=[0.5])\n";
    script << "\n";
    script << "    # Визуализация точек данных\n";
    script << "    if not points_df.empty:\n";
    script << "        # Разделяем точки по классам\n";
    script << "        for label in points_df['label'].unique():\n";
    script << "            class_points = points_df[points_df['label'] == label]\n";
    script << "            ax.scatter(class_points['x'], class_points['y'], \n";
    script << "                      label=f'Class {label}', s=50, alpha=0.7, edgecolors='black')\n";
    script << "\n";
    script << "    # Настройка графика\n";
    script << "    ax.set_xlabel('Feature 0', fontsize=12)\n";
    script << "    ax.set_ylabel('Feature 1', fontsize=12)\n";
    script << "    ax.set_title('2D Visualization with Decision Boundary', fontsize=14)\n";
    script << "    ax.legend()\n";
    script << "    ax.grid(True, alpha=0.3)\n";
    script << "\n";
    script << "    # Сохранение графика\n";
    script << "    plt.tight_layout()\n";
    script << "    plt.savefig('" << outputImage << "', dpi=300, bbox_inches='tight')\n";
    script << "    print(f'График сохранен как {outputImage}')\n";
    script << "    \n";
    script << "    # Показать график (если запущено в интерактивном режиме)\n";
    script << "    if 'DISPLAY' in os.environ:\n";
    script << "        plt.show()\n";
    script << "\n";
    script << "if __name__ == '__main__':\n";
    script << "    main()\n";
    
    script.close();
    
    // Делаем скрипт исполняемым (для Unix-систем)
    #ifdef _WIN32
        // На Windows не нужно менять права
    #else
        system(("chmod +x visualize_2d.py 2>/dev/null"));
    #endif
}

// Генерация Python скрипта для истории обучения
void Visualizer::generateTrainingHistoryScript(
    const std::string& historyFile,
    const std::string& outputImage) {
    
    ofstream script("visualize_history.py");
    if (!script.is_open()) {
        cerr << "Visualizer::generateTrainingHistoryScript: Cannot create Python script" << endl;
        return;
    }
    
    script << "#!/usr/bin/env python3\n";
    script << "# Автоматически сгенерированный скрипт для визуализации истории обучения\n";
    script << "\n";
    script << "import pandas as pd\n";
    script << "import matplotlib.pyplot as plt\n";
    script << "import numpy as np\n";
    script << "import sys\n";
    script << "import os\n";
    script << "\n";
    script << "def main():\n";
    script << "    # Загрузка данных\n";
    script << "    try:\n";
    script << "        history_df = pd.read_csv('" << historyFile << "')\n";
    script << "    except FileNotFoundError as e:\n";
    script << "        print(f'Ошибка: Файл не найден: {e}')\n";
    script << "        sys.exit(1)\n";
    script << "\n";
    script << "    # Создание графика\n";
    script << "    fig, ax = plt.subplots(figsize=(10, 6))\n";
    script << "\n";
    script << "    # Проверяем наличие столбцов\n";
    script << "    if 'generation' in history_df.columns and 'loss' in history_df.columns:\n";
    script << "        ax.plot(history_df['generation'], history_df['loss'], \n";
    script << "                'b-', linewidth=2, label='Loss')\n";
    script << "    \n";
    script << "    if 'generation' in history_df.columns and 'best_loss' in history_df.columns:\n";
    script << "        ax.plot(history_df['generation'], history_df['best_loss'], \n";
    script << "                'r--', linewidth=2, label='Best Loss')\n";
    script << "\n";
    script << "    # Настройка графика\n";
    script << "    ax.set_xlabel('Generation', fontsize=12)\n";
    script << "    ax.set_ylabel('Loss', fontsize=12)\n";
    script << "    ax.set_title('Training History', fontsize=14)\n";
    script << "    ax.legend()\n";
    script << "    ax.grid(True, alpha=0.3)\n";
    script << "    \n";
    script << "    # Логарифмическая шкала по оси Y, если значения сильно различаются\n";
    script << "    if 'loss' in history_df.columns:\n";
    script << "        loss_values = history_df['loss'].dropna().values\n";
    script << "        if len(loss_values) > 0 and (loss_values.max() / loss_values.min() > 100):\n";
    script << "            ax.set_yscale('log')\n";
    script << "\n";
    script << "    # Сохранение графика\n";
    script << "    plt.tight_layout()\n";
    script << "    plt.savefig('" << outputImage << "', dpi=300, bbox_inches='tight')\n";
    script << "    print(f'График сохранен как {outputImage}')\n";
    script << "    \n";
    script << "    # Показать график (если запущено в интерактивном режиме)\n";
    script << "    if 'DISPLAY' in os.environ:\n";
    script << "        plt.show()\n";
    script << "\n";
    script << "if __name__ == '__main__':\n";
    script << "    main()\n";
    
    script.close();
    
    // Делаем скрипт исполняемым (для Unix-систем)
    #ifdef _WIN32
        // На Windows не нужно менять права
    #else
        system(("chmod +x visualize_history.py 2>/dev/null"));
    #endif
}

// Основные методы визуализации

void Visualizer::plot2D(const DataSet& data, const NeuralNetwork& net, 
                       const std::string& outputPath) {
    
    if (data.empty()) {
        cerr << "Visualizer::plot2D: Dataset is empty" << endl;
        return;
    }
    
    // Проверяем, что данные 2D
    const DataPoint& firstPoint = data.get(0);
    if (firstPoint.getFeatures().size() != 2) {
        cerr << "Visualizer::plot2D: Data must be 2D for visualization. Found " 
             << firstPoint.getFeatures().size() << " dimensions." << endl;
        return;
    }
    
    cout << "Generating 2D visualization..." << endl;
    
    // Создаем имена файлов
    string pointsFile = outputPath + "_points.csv";
    string boundaryFile = outputPath + "_boundary.csv";
    string scriptFile = "visualize_2d.py";
    
    // Сохраняем точки данных
    savePointsToCSV(data, pointsFile);
    cout << "  Saved points data to: " << pointsFile << endl;
    
    // Генерируем сетку и сохраняем границу решения
    vector<pair<double, double>> grid = generateGrid(data, 100, 0.2);
    if (!grid.empty()) {
        saveDecisionBoundaryToCSV(net, grid, boundaryFile);
        cout << "  Saved decision boundary to: " << boundaryFile << endl;
    } else {
        cerr << "  Warning: Could not generate grid for decision boundary" << endl;
    }
    
    // Генерируем Python скрипт для визуализации
    generateVisualizationScript(pointsFile, boundaryFile, outputPath);
    cout << "  Generated visualization script: " << scriptFile << endl;
    
    // Запускаем Python скрипт (опционально)
    cout << "  To generate the plot, run: python3 " << scriptFile << endl;
    cout << "  Or manually execute the generated script." << endl;
}

void Visualizer::plotMultiDimensional(const DataSet& data, const NeuralNetwork& net,
                                    const std::string& outputPath) {
    // Заглушка для будущей реализации
    cerr << "Visualizer::plotMultiDimensional: Not implemented yet" << endl;
    
    // Создаем простой скрипт для PCA визуализации
    ofstream script("visualize_pca.py");
    if (script.is_open()) {
        script << "#!/usr/bin/env python3\n";
        script << "print('PCA visualization not implemented yet. Need to implement data export first.')\n";
        script.close();
        cout << "  Generated placeholder script: visualize_pca.py" << endl;
    }
}

void Visualizer::plotTrainingHistory(const std::vector<double>& history,
                                   const std::string& outputPath) {
    
    if (history.empty()) {
        cerr << "Visualizer::plotTrainingHistory: History is empty" << endl;
        return;
    }
    
    cout << "Generating training history visualization..." << endl;
    
    // Сохраняем историю обучения в CSV
    string historyFile = outputPath + "_history.csv";
    ofstream file(historyFile);
    
    if (!file.is_open()) {
        cerr << "  Cannot open file: " << historyFile << endl;
        return;
    }
    
    file << "generation,loss" << endl;
    for (size_t i = 0; i < history.size(); ++i) {
        file << i << "," << history[i] << endl;
    }
    file.close();
    
    cout << "  Saved training history to: " << historyFile << endl;
    
    // Генерируем Python скрипт для визуализации
    generateTrainingHistoryScript(historyFile, outputPath);
    cout << "  Generated visualization script: visualize_history.py" << endl;
    cout << "  To generate the plot, run: python3 visualize_history.py" << endl;
}