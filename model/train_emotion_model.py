# train_emotion_models.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import cv2
import warnings
import pickle

warnings.filterwarnings('ignore')

# Константы
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30  # ← ИЗМЕНЕНО С 50 НА 30
CLASS_NAMES = ['Злость', 'Радость', 'Грусть', 'Удивление']


class EmotionModelTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train_path = os.path.join(dataset_path, "train")
        self.test_path = os.path.join(dataset_path, "test")

        # Маппинг английских названий папок на русские классы
        self.class_mapping = {
            'angry': 'Злость',
            'happy': 'Радость',
            'sad': 'Грусть',
            'surprise': 'Удивление'
        }

    def prepare_data(self):
        """Подготовка данных с аугментацией"""

        print(f"Путь к train: {self.train_path}")
        print(f"Путь к test: {self.test_path}")

        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            print("ОШИБКА: Папки train или test не найдены!")
            return None, None

        # Аугментация для тренировочных данных
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,  # Уменьшил для скорости
            width_shift_range=0.1,  # Уменьшил
            height_shift_range=0.1,  # Уменьшил
            shear_range=0.1,  # Уменьшил
            zoom_range=0.1,  # Уменьшил
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Для тестовых данных только нормализация
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        try:
            train_generator = train_datagen.flow_from_directory(
                self.train_path,
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                shuffle=True,
                classes=list(self.class_mapping.keys())
            )

            test_generator = test_datagen.flow_from_directory(
                self.test_path,
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                shuffle=False,
                classes=list(self.class_mapping.keys())
            )

            print(f"\nНайдено классов: {train_generator.num_classes}")
            print(f"Тренировочных образцов: {train_generator.samples}")
            print(f"Тестовых образцов: {test_generator.samples}")
            print(f"Эпох обучения: {EPOCHS}")  # Добавил вывод количества эпох

            return train_generator, test_generator

        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return None, None

    def create_resnet_model(self):
        """Создание модели на основе ResNet50"""
        base_model = applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),  # Уменьшил с 256 до 128
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),  # Уменьшил с 128 до 64
            layers.Dropout(0.2),
            layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])

        return model, base_model

    def create_vgg_model(self):
        """Создание модели на основе VGG16"""
        base_model = applications.VGG16(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),  # Уменьшил с 512 до 256
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),  # Уменьшил с 256 до 128
            layers.Dropout(0.3),
            layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])

        return model, base_model

    def create_mobilenet_model(self):
        """Создание модели на основе MobileNetV2"""
        base_model = applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),  # Уменьшил с 128 до 64
            layers.Dropout(0.2),
            layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])

        return model, base_model

    def plot_training_history(self, history, model_name):
        """Визуализация процесса обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # График точности
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'{model_name} - Точность модели ({EPOCHS} эпох)')
        axes[0].set_xlabel('Эпохи')
        axes[0].set_ylabel('Точность')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xticks(range(0, EPOCHS, 5))  # Отмечаем каждые 5 эпох

        # График потерь
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'{model_name} - Потери модели ({EPOCHS} эпох)')
        axes[1].set_xlabel('Эпохи')
        axes[1].set_ylabel('Потери')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_xticks(range(0, EPOCHS, 5))  # Отмечаем каждые 5 эпох

        plt.tight_layout()
        plt.savefig(f'{model_name}_history.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix_percent(self, model, test_generator, model_name):
        """Построение матрицы ошибок в процентах"""

        test_generator.reset()
        predictions = model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes

        # Строим матрицу ошибок
        cm = confusion_matrix(y_true, y_pred)

        # Преобразуем в проценты
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Визуализация
        plt.figure(figsize=(12, 8))

        # Создаем heatmap с процентами
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    vmin=0, vmax=100)

        plt.title(f'{model_name} - Матрица ошибок (%) - {EPOCHS} эпох', fontsize=16, pad=20)
        plt.ylabel('Истинные значения', fontsize=12)
        plt.xlabel('Предсказанные значения', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Сохраняем матрицу в текстовом формате
        with open(f'{model_name}_confusion_matrix.txt', 'w', encoding='utf-8') as f:
            f.write(f"Матрица ошибок для {model_name} (в %) - {EPOCHS} эпох\n")
            f.write("-" * 50 + "\n")
            f.write("   " + " ".join([f"{c:8}" for c in CLASS_NAMES]) + "\n")
            for i, row in enumerate(cm_percent):
                f.write(f"{CLASS_NAMES[i]:8} " + " ".join([f"{val:8.1f}" for val in row]) + "\n")

        print(f"\nМатрица ошибок для {model_name} сохранена")

        # Детальный отчет
        print(f"\n{'=' * 50}")
        print(f"ОТЧЕТ ПО КЛАССАМ - {model_name} ({EPOCHS} эпох)")
        print(f"{'=' * 50}")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

        return cm_percent

    def train_model(self, model_name, create_model_func):
        """Обучение конкретной модели"""

        print(f"\n{'=' * 60}")
        print(f"ОБУЧЕНИЕ МОДЕЛИ: {model_name}")
        print(f"Количество эпох: {EPOCHS}")
        print(f"{'=' * 60}")

        # Подготовка данных
        print("\n[1/5] Подготовка данных...")
        train_generator, test_generator = self.prepare_data()

        if train_generator is None or test_generator is None:
            print("Ошибка подготовки данных")
            return None

        # Создание модели
        print(f"\n[2/5] Создание модели {model_name}...")
        model, base_model = create_model_func()

        # Компиляция
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                f'{model_name}_best.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,  # Уменьшил с 10 до 5 для 30 эпох
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,  # Уменьшил с 5 до 3
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Обучение
        print(f"\n[3/5] Начало обучения {model_name} на {EPOCHS} эпохах...")
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_data=test_generator,
            validation_steps=test_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        # Загружаем лучшую модель
        model = keras.models.load_model(f'{model_name}_best.keras')

        # Визуализация
        print(f"\n[4/5] Построение графиков обучения...")
        self.plot_training_history(history, model_name)

        # Матрица ошибок
        print(f"\n[5/5] Построение матрицы ошибок...")
        cm_percent = self.plot_confusion_matrix_percent(model, test_generator, model_name)

        # Сохраняем историю
        with open(f'{model_name}_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        print(f"\n{'=' * 60}")
        print(f"ОБУЧЕНИЕ {model_name} ЗАВЕРШЕНО!")
        print(f"Лучшая точность: {max(history.history['val_accuracy']):.3f}")
        print(f"{'=' * 60}")

        return {
            'model': model,
            'history': history,
            'confusion_matrix': cm_percent,
            'name': model_name
        }

    def train_all_models(self):
        """Обучение всех трех моделей"""

        models_config = [
            ('resnet50', self.create_resnet_model),
            ('vgg16', self.create_vgg_model),
            ('mobilenetv2', self.create_mobilenet_model)
        ]

        results = []
        for model_name, create_func in models_config:
            print(f"\n{'#' * 70}")
            print(f"#" + f" НАЧАЛО ОБУЧЕНИЯ {model_name.upper()} ".center(68) + "#")
            print(f"{'#' * 70}")

            result = self.train_model(model_name, create_func)
            if result:
                results.append(result)

            # Очистка сессии TensorFlow для экономии памяти
            tf.keras.backend.clear_session()

        # Сравнение моделей
        self.compare_models(results)

        return results

    def compare_models(self, results):
        """Сравнение всех моделей"""

        if not results:
            return

        plt.figure(figsize=(15, 6))

        # Сравнение точности
        plt.subplot(1, 2, 1)
        for result in results:
            plt.plot(result['history'].history['val_accuracy'],
                     label=f"{result['name']} (best: {max(result['history'].history['val_accuracy']):.3f})")
        plt.title(f'Сравнение точности моделей ({EPOCHS} эпох)')
        plt.xlabel('Эпохи')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(0, EPOCHS, 5))

        # Сравнение потерь
        plt.subplot(1, 2, 2)
        for result in results:
            plt.plot(result['history'].history['val_loss'],
                     label=f"{result['name']} (min: {min(result['history'].history['val_loss']):.3f})")
        plt.title(f'Сравнение потерь моделей ({EPOCHS} эпох)')
        plt.xlabel('Эпохи')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(0, EPOCHS, 5))

        plt.tight_layout()
        plt.savefig('models_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Сохраняем результаты сравнения
        with open('models_comparison.txt', 'w', encoding='utf-8') as f:
            f.write(f"СРАВНЕНИЕ МОДЕЛЕЙ ({EPOCHS} эпох)\n")
            f.write("=" * 50 + "\n\n")
            for result in results:
                val_acc = max(result['history'].history['val_accuracy'])
                val_loss = min(result['history'].history['val_loss'])
                f.write(f"{result['name']}:\n")
                f.write(f"  Лучшая точность: {val_acc:.3f}\n")
                f.write(f"  Минимальные потери: {val_loss:.3f}\n\n")


def main():
    # Укажите путь к вашему датасету
    DATASET_PATH = "C:/Users/Tamerlan/PycharmProjects/Human_Emotion_Detections_CV/dataset"

    trainer = EmotionModelTrainer(DATASET_PATH)
    results = trainer.train_all_models()

    print("\n" + "=" * 60)
    print(f"ВСЕ МОДЕЛИ УСПЕШНО ОБУЧЕНЫ! ({EPOCHS} эпох)")
    print("=" * 60)
    print("\nСохраненные файлы:")
    print("  - resnet50_best.keras")
    print("  - vgg16_best.keras")
    print("  - mobilenetv2_best.keras")
    print("  - *_history.png (графики обучения)")
    print("  - *_confusion_matrix.png (матрицы ошибок)")
    print("  - models_comparison.png (сравнение моделей)")


if __name__ == "__main__":
    main()