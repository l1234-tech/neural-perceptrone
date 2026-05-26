import nnlib
import random

# Создание нейронной сети
network = nnlib.create_network([2, 3, 1])  # 2 входа, 3 нейрона в скрытом слое, 1 выход

# Загрузка данных
dataset = nnlib.load_dataset("data/train.csv")

# Создание тренера
trainer = nnlib.GradientTrainer(learning_rate=0.1, epochs=100)

# Обучение
history = trainer.train(network, dataset)

# Предсказание
prediction = network.predict([0.5, -0.2])
print(f"Prediction: {prediction}")

# Дообучение на новых данных (fine-tuning)
new_data = nnlib.load_dataset("data/d1.csv")
history_continued = trainer.continue_training(network, new_data)

# Оценка метрик
metrics = nnlib.calculate_metrics(network, dataset)
print(f"Accuracy: {metrics.accuracy}, Weighted F1: {metrics.weighted_f1}")