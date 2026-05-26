import nnlib
import random
import sys
import io

# Принудительно устанавливаем UTF-8 для stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Создание нейронной сети
network_1 = nnlib.create_network([2, 3, 1])  # 2 входа, 3 нейрона в скрытом слое, 1 выход
network_2 = nnlib.create_network([4, 3, 1])  # 4 входа, 3 нейрона в скрытом слое, 1 выход
# Загрузка данных
dataset = nnlib.load_dataset("data/d1.csv")

# Создание тренера
trainer = nnlib.GradientTrainer(learning_rate=0.1, epochs=100)

# Обучение
history = trainer.train(network_1, dataset)

# Предсказание
prediction = network_1.predict([0.5, -0.2])
print(f"Prediction: {prediction}")

# Дообучение на новых данных (fine-tuning)
new_data = nnlib.load_dataset("data/d2.csv")
history_continued = trainer.continue_training(network_2, new_data)

# Оценка метрик
metrics_1 = nnlib.calculate_metrics(network_1, dataset)
print(f"Accuracy: {metrics_1.accuracy}, Weighted F1: {metrics_1.weighted_f1}")

metrics_2 = nnlib.calculate_metrics(network_2, new_data)
print(f"Accuracy: {metrics_2.accuracy}, Weighted F1: {metrics_2.weighted_f1}")