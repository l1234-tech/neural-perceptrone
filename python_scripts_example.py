import nnlib
import sys
import io

# Принудительно устанавливаем UTF-8 для stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def train_and_evaluate(network, dataset, trainer, label="Model"):
    """Обучает сеть, делает тестовое предсказание и выводит метрики."""
    # Если это первое обучение, используем train, иначе continue_training
    # В данном случае, так как сети новые, всегда используем train для чистоты,
    # но если нужно именно дообучение ранее использованной сети, логика чуть изменится.
    # Исходя из вашего кода, network_1 обучалась с нуля, остальные тоже создавались заново.
    trainer.train(network, dataset)

    metrics = nnlib.calculate_metrics(network, dataset)
    print(f"{label} -> Accuracy: {metrics.accuracy:.4f}, Weighted F1: {metrics.weighted_f1:.4f}")

# Инициализация тренера (общий для всех, так как параметры одинаковые)
trainer = nnlib.GradientTrainer(learning_rate=0.1, epochs=1000)

# Конфигурация сетей и данных: (архитектура, путь к данным, имя для вывода)
configs = [
    ([2, 3, 1], "data/train.csv", "Network_Train"),
    ([2, 3, 1], "data/d1.csv", "Network_1"),
    ([4, 3, 1], "data/d2.csv", "Network_2"),
]

for architecture, data_path, name in configs:
    network = nnlib.create_network(architecture)
    dataset = nnlib.load_dataset(data_path)
    train_and_evaluate(network, dataset, trainer, label=name)