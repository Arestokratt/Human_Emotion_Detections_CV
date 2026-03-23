import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import pickle
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Константы
BATCH_SIZE = 32
CLASS_NAMES = ['Злость', 'Радость', 'Грусть', 'Удивление']


class EnsembleEvaluator:
    def __init__(self, dataset_path, models_dir="C:/Users/Tamerlan/PycharmProjects/Human_Emotion_Detections_CV/model"):
        self.dataset_path = dataset_path
        self.models_dir = models_dir
        self.test_path = os.path.join(dataset_path, "test")

        # Маппинг папок
        self.class_mapping = {
            'angry': 'Злость',
            'happy': 'Радость',
            'sad': 'Грусть',
            'surprise': 'Удивление'
        }

        # Размеры для разных моделей
        self.model_sizes = {
            'Xception': (299, 299),
            'VGG16': (224, 224),
            'MobileNetV2': (224, 224)
        }

        self.models = {}

    def load_models(self):
        """Загрузка всех обученных моделей"""
        print("\n" + "=" * 60)
        print("ЗАГРУЗКА МОДЕЛЕЙ ДЛЯ АНСАМБЛЯ")
        print("=" * 60)

        models_to_load = [
            ('xception_best.keras', 'Xception'),
            ('vgg16_best.keras', 'VGG16'),
            ('mobilenetv2_best.keras', 'MobileNetV2')
        ]

        loaded_count = 0
        for model_file, model_name in models_to_load:
            model_path = os.path.join(self.models_dir, model_file)
            try:
                if os.path.exists(model_path):
                    model = keras.models.load_model(model_path)
                    self.models[model_name] = {
                        'model': model,
                        'size': self.model_sizes[model_name]
                    }
                    loaded_count += 1
                    print(f"✓ Загружена: {model_name}")
                else:
                    print(f"✗ Файл {model_file} не найден")
            except Exception as e:
                print(f"✗ Ошибка загрузки {model_name}: {e}")

        print(f"\nЗагружено моделей: {loaded_count}/3")
        return loaded_count == 3

    def prepare_data(self):
        """Подготовка тестовых данных"""
        print("\n[1/3] Подготовка данных...")

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        try:
            test_generator = test_datagen.flow_from_directory(
                self.test_path,
                target_size=(224, 224),  # Временный размер, будет меняться
                batch_size=1,  # По одному для точности
                class_mode='categorical',
                shuffle=False,
                classes=list(self.class_mapping.keys())
            )

            print(f"✓ Тестовых образцов: {test_generator.samples}")
            return test_generator

        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return None

    def predict_with_ensemble(self, image, models_dict):
        """Предсказание ансамблем моделей"""
        predictions = {}

        for model_name, model_info in models_dict.items():
            model = model_info['model']
            target_size = model_info['size']

            # Изменяем размер под конкретную модель
            img_resized = tf.image.resize(image, target_size)
            img_array = np.expand_dims(img_resized.numpy(), axis=0)

            # Предсказание
            preds = model.predict(img_array, verbose=0)[0]
            pred_class = np.argmax(preds)

            predictions[model_name] = pred_class

        # Голосование
        votes = list(predictions.values())
        most_common = Counter(votes).most_common(1)[0][0]

        return most_common, predictions

    def evaluate_ensemble(self, test_generator):
        """Оценка ансамбля на тестовых данных"""

        print("\n[2/3] Оценка ансамбля моделей...")

        # Сбрасываем генератор
        test_generator.reset()

        # Для хранения результатов
        y_true = []
        y_pred_ensemble = []
        y_pred_individual = {
            'Xception': [],
            'VGG16': [],
            'MobileNetV2': []
        }

        total_samples = test_generator.samples
        processed = 0

        print("Обработка изображений...")

        for i in range(total_samples):
            # Получаем изображение и метку
            img, label = test_generator[i]
            img = img[0]  # Убираем batch dimension
            true_class = np.argmax(label[0])

            # Предсказание ансамблем
            pred_ensemble, pred_individual = self.predict_with_ensemble(img, self.models)

            # Сохраняем результаты
            y_true.append(true_class)
            y_pred_ensemble.append(pred_ensemble)

            for model_name in y_pred_individual.keys():
                y_pred_individual[model_name].append(pred_individual[model_name])

            processed += 1
            if processed % 50 == 0:
                print(f"  Обработано: {processed}/{total_samples}")

        print(f"✓ Обработано: {processed}/{total_samples}")

        return y_true, y_pred_ensemble, y_pred_individual

    def plot_confusion_matrix_percent(self, y_true, y_pred, title, filename):
        """Построение матрицы ошибок в процентах"""

        # Строим матрицу
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Визуализация
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    vmin=0, vmax=100)

        plt.title(f'{title} - Матрица ошибок (%)', fontsize=16, pad=20)
        plt.ylabel('Истинные значения', fontsize=12)
        plt.xlabel('Предсказанные значения', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()

        # Сохраняем текстовую версию
        txt_filename = filename.replace('.png', '.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"{title} - Матрица ошибок (в %)\n")
            f.write("-" * 50 + "\n")
            f.write("   " + " ".join([f"{c:8}" for c in CLASS_NAMES]) + "\n")
            for i, row in enumerate(cm_percent):
                f.write(f"{CLASS_NAMES[i]:8} " + " ".join([f"{val:8.1f}" for val in row]) + "\n")

        return cm_percent

    def print_classification_report(self, y_true, y_pred, title):
        """Вывод отчета по классам"""
        print(f"\n{'=' * 60}")
        print(f"{title} - ОТЧЕТ ПО КЛАССАМ")
        print(f"{'=' * 60}")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    def compare_with_individual(self, y_true, y_pred_ensemble, y_pred_individual):
        """Сравнение ансамбля с отдельными моделями"""

        print("\n[3/3] Сравнение результатов...")

        # Сохраняем метрики
        metrics = {}

        # Ансамбль
        from sklearn.metrics import accuracy_score
        ensemble_acc = accuracy_score(y_true, y_pred_ensemble)
        metrics['Ensemble'] = ensemble_acc

        # Отдельные модели
        for model_name in y_pred_individual.keys():
            acc = accuracy_score(y_true, y_pred_individual[model_name])
            metrics[model_name] = acc

        # Вывод сравнения
        print(f"\n{'=' * 60}")
        print("СРАВНЕНИЕ ТОЧНОСТИ")
        print(f"{'=' * 60}")
        for model_name, acc in sorted(metrics.items(), key=lambda x: x[1], reverse=True):
            print(f"{model_name:15} : {acc:.3f}")

        # Создаем график сравнения
        plt.figure(figsize=(10, 6))
        models = list(metrics.keys())
        accuracies = list(metrics.values())

        colors = ['#2ecc71' if m == 'Ensemble' else '#3498db' for m in models]
        bars = plt.bar(models, accuracies, color=colors)
        plt.ylim(0, 1)
        plt.ylabel('Точность', fontsize=12)
        plt.title('Сравнение точности: Ансамбль vs Отдельные модели', fontsize=14)
        plt.xticks(rotation=45)

        # Добавляем значения на столбцы
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('models/ensemble_comparison.png', dpi=150)
        plt.show()

        # Сохраняем результаты
        with open('models/ensemble_results.txt', 'w', encoding='utf-8') as f:
            f.write("РЕЗУЛЬТАТЫ АНСАМБЛЯ МОДЕЛЕЙ\n")
            f.write("=" * 60 + "\n\n")

            f.write("Точность моделей:\n")
            for model_name, acc in sorted(metrics.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {model_name}: {acc:.3f}\n")

            f.write(f"\nАнсамбль улучшил результат на:\n")
            best_individual = max([acc for name, acc in metrics.items() if name != 'Ensemble'])
            improvement = ensemble_acc - best_individual
            f.write(f"  {improvement:.3f} ({improvement * 100:.1f}%) по сравнению с лучшей моделью\n")

        return metrics

    def run(self):
        """Запуск оценки ансамбля"""

        print("\n" + "=" * 60)
        print("ОЦЕНКА АНСАМБЛЯ ИЗ 3 МОДЕЛЕЙ")
        print("=" * 60)

        # 1. Загружаем модели
        if not self.load_models():
            print("Ошибка: Не удалось загрузить все модели")
            return

        # 2. Подготавливаем данные
        test_generator = self.prepare_data()
        if test_generator is None:
            return

        # 3. Оцениваем ансамбль
        y_true, y_pred_ensemble, y_pred_individual = self.evaluate_ensemble(test_generator)

        # 4. Строим матрицу ошибок для ансамбля
        print("\nПостроение матрицы ошибок для ансамбля...")
        self.plot_confusion_matrix_percent(
            y_true, y_pred_ensemble,
            "Ensemble (Голосование 3 моделей)",
            "models/ensemble_confusion_matrix.png"
        )
        self.print_classification_report(y_true, y_pred_ensemble, "ENSEMBLE")

        # 5. Строим матрицы для отдельных моделей
        print("\nПостроение матриц ошибок для отдельных моделей...")
        for model_name in y_pred_individual.keys():
            self.plot_confusion_matrix_percent(
                y_true, y_pred_individual[model_name],
                f"{model_name}",
                f"models/{model_name.lower()}_confusion_matrix_ensemble.png"
            )
            self.print_classification_report(y_true, y_pred_individual[model_name], model_name.upper())

        # 6. Сравниваем результаты
        metrics = self.compare_with_individual(y_true, y_pred_ensemble, y_pred_individual)

        # Итоговый вывод
        print("\n" + "=" * 60)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 60)
        print(f"\n✓ Ансамбль (голосование 3 моделей) показал точность: {metrics['Ensemble']:.3f}")
        print(f"✓ Результаты сохранены в папке 'models/':")
        print("  - ensemble_confusion_matrix.png (матрица ошибок ансамбля)")
        print("  - xception_confusion_matrix_ensemble.png")
        print("  - vgg16_confusion_matrix_ensemble.png")
        print("  - mobilenetv2_confusion_matrix_ensemble.png")
        print("  - ensemble_comparison.png (сравнение)")
        print("  - ensemble_results.txt (текстовый отчет)")

        return metrics


def main():
    DATASET_PATH = "C:/Users/Tamerlan/PycharmProjects/Human_Emotion_Detections_CV/dataset"

    # Проверяем наличие папки models
    if not os.path.exists('models'):
        os.makedirs('models')

    # Запускаем оценку ансамбля
    evaluator = EnsembleEvaluator(DATASET_PATH)
    results = evaluator.run()

    if results:
        print("\n" + "=" * 60)
        print("ГОТОВО! МОЖНО ЗАПУСКАТЬ MAIN.PY")
        print("=" * 60)
        print("\nТеперь в приложении будет доступно голосование")
        print("из трех моделей для более точного результата!")


if __name__ == "__main__":
    main()
