import nnlib
import sys
import io

# Принудительно устанавливаем UTF-8 для stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def train_and_evaluate(network, dataset, trainer, label="Model"):
    """Обучает сеть, делает тестовое предсказание и выводит метрики."""
    trainer.train(network, dataset)
    metrics = nnlib.calculate_metrics(network, dataset)
    print(f"{label} -> Accuracy: {metrics.accuracy:.4f}, Weighted F1: {metrics.weighted_f1:.4f}")


MODEL_TYPE = "gradient"  # Измените на "genetic" для генетического алгоритма

# Параметры для GradientTrainer (используются если MODEL_TYPE = "gradient")
GRADIENT_PARAMS = {
    "learning_rate": 0.1,
    "epochs": 1000,
    "batch_size": 32,
    "lambda_param": 0.01,  
    "early_stopping_patience": 10,
    "validation_split": 0.2,
    "shuffle": True
}

# Параметры для GeneticTrainer (используются если MODEL_TYPE = "genetic")
GENETIC_PARAMS = {
    "population_size": 50,
    "max_generations": 100,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "elitism_ratio": 0.1,
    "regularization_lambda": 0.01,
    "early_stopping_patience": 10,
    "use_f1_fitness": True
}
# ===== КОНЕЦ НАСТРОЕК =====

def create_trainer(model_type, gradient_params, genetic_params):
    """Создает тренера на основе выбранного типа модели."""
    if model_type == "genetic":
        trainer = nnlib.GeneticTrainer(
            population_size=genetic_params["population_size"],
            max_generations=genetic_params["max_generations"],
            mutation_rate=genetic_params["mutation_rate"],
            crossover_rate=genetic_params["crossover_rate"],
            elitism_ratio=genetic_params["elitism_ratio"],
            regularization_lambda=genetic_params["regularization_lambda"],
            early_stopping_patience=genetic_params["early_stopping_patience"],
            use_f1_fitness=genetic_params["use_f1_fitness"]
        )
        print(f"Используется GeneticTrainer с параметрами:")
        for key, value in genetic_params.items():
            print(f"  {key}: {value}")
    else:
        # GradientTrainer - передаем позиционные аргументы в правильном порядке
        # Порядок: learning_rate, epochs, batch_size, lambda, early_stopping_patience, validation_split, shuffle
        # Используем lambda_param из словаря, но передаем как позиционный аргумент
        trainer = nnlib.GradientTrainer(
            gradient_params["learning_rate"],
            gradient_params["epochs"],
            gradient_params["batch_size"],
            gradient_params["lambda_param"],  # 4-й параметр - lambda
            gradient_params["early_stopping_patience"],
            gradient_params["validation_split"],
            gradient_params["shuffle"]
        )
        print(f"Используется GradientTrainer с параметрами:")
        # Выводим параметры, заменяя lambda_param на lambda для ясности
        params_display = gradient_params.copy()
        params_display["lambda"] = params_display.pop("lambda_param")
        for key, value in params_display.items():
            print(f"  {key}: {value}")
    return trainer

def main():
    print("=" * 60)
    print("Запуск обучения нейронных сетей")
    print(f"Выбранная модель: {MODEL_TYPE}")
    print("=" * 60)
    
    # Создаем тренера
    trainer = create_trainer(MODEL_TYPE, GRADIENT_PARAMS, GENETIC_PARAMS)
    
    # Конфигурация сетей и данных: (архитектура, путь к данным, имя для вывода)
    configs = [
        # ([2, 3, 1], "data/train.csv", "Network_Train"),
        ([2, 3, 1], "data/d1.csv", "Network_1"),
        ([4, 3, 1], "data/d2.csv", "Network_2"),
        # ([n, 3, 1], "data/d3.csv", "Network_d3"),
    ]
    
    for architecture, data_path, name in configs:
        network = nnlib.create_network(architecture)
        dataset = nnlib.load_dataset(data_path)
        train_and_evaluate(network, dataset, trainer, label=name)
    
    print("=" * 60)
    print("Обучение завершено для всех конфигураций")
    print("=" * 60)

if __name__ == "__main__":
    main()