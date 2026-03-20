# train_efficientnet.py
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
import warnings
import pickle

warnings.filterwarnings('ignore')

# Константы
IMG_SIZE = (224, 224)  # EfficientNetV2-S использует 224x224
BATCH_SIZE = 32
EPOCHS = 30
CLASS_NAMES = ['Злость', 'Радость', 'Грусть', 'Удивление']


class EfficientNetTrainer:
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
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
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

            return train_generator, test_generator

        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return None, None

    def create_efficientnet_model(self):
        """Создание модели EfficientNetV2-S"""
        print("\n" + "=" * 50)
        print("СОЗДАНИЕ МОДЕЛИ EfficientNetV2-S")
        print("=" * 50)

        try:
            # Пробуем загрузить EfficientNetV2S
            base_model = applications.EfficientNetV2S(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            model_name = "efficientnetv2s"
            print("✓ Загружен EfficientNetV2-S")

        except Exception as e:
            print(f"EfficientNetV2-S не доступен: {e}")
            print("Пробуем EfficientNetB0...")

            # Запасной вариант - EfficientNetB0
            base_model = applications.EfficientNetB0(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            model_name = "efficientnetb0"
            print("✓ Загружен EfficientNetB0 (альтернатива)")

        # Замораживаем базовую модель
        base_model.trainable = False
        print(f"✓ Всего слоев в базовой модели: {len(base_model.layers)}")

        # Создаем полную модель
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(CLASS_NAMES), activation='softmax')
        ], name=model_name)

        # Компиляция
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("✓ Модель успешно создана и скомпилирована")
        model.summary()

        return model, base_model, model_name

    def plot_training_history(self, history, model_name):
        """Визуализация процесса обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # График точности
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'{model_name} - Точность модели')
        axes[0].set_xlabel('Эпохи')
        axes[0].set_ylabel('Точность')
        axes[0].legend()
        axes[0].grid(True)

        # График потерь
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'{model_name} - Потери модели')
        axes[1].set_xlabel('Эпохи')
        axes[1].set_ylabel('Потери')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'models/{model_name}_history.png', dpi=150, bbox_inches='tight')
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

        plt.title(f'{model_name} - Матрица ошибок (%)', fontsize=16, pad=20)
        plt.ylabel('Истинные значения', fontsize=12)
        plt.xlabel('Предсказанные значения', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'models/{model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Сохраняем матрицу в текстовом формате
        with open(f'models/{model_name}_confusion_matrix.txt', 'w', encoding='utf-8') as f:
            f.write(f"Матрица ошибок для {model_name} (в %)\n")
            f.write("-" * 50 + "\n")
            f.write("   " + " ".join([f"{c:8}" for c in CLASS_NAMES]) + "\n")
            for i, row in enumerate(cm_percent):
                f.write(f"{CLASS_NAMES[i]:8} " + " ".join([f"{val:8.1f}" for val in row]) + "\n")

        # Детальный отчет
        print(f"\n{'=' * 50}")
        print(f"ОТЧЕТ ПО КЛАССАМ - {model_name}")
        print(f"{'=' * 50}")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

        return cm_percent

    def train_model(self):
        """Обучение только EfficientNet модели"""

        print(f"\n{'=' * 60}")
        print("ОБУЧЕНИЕ МОДЕЛИ EFFICIENTNET")
        print(f"{'=' * 60}")

        # Подготовка данных
        print("\n[1/5] Подготовка данных...")
        train_generator, test_generator = self.prepare_data()

        if train_generator is None or test_generator is None:
            print("Ошибка подготовки данных")
            return None

        # Создание модели
        print(f"\n[2/5] Создание модели...")
        model, base_model, model_name = self.create_efficientnet_model()

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                f'models/{model_name}_best.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Обучение
        print(f"\n[3/5] Начало обучения {model_name}...")
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
        best_model_path = f'models/{model_name}_best.keras'
        if os.path.exists(best_model_path):
            model = keras.models.load_model(best_model_path)
            print(f"✓ Загружена лучшая модель из {best_model_path}")

        # Визуализация
        print(f"\n[4/5] Построение графиков обучения...")
        self.plot_training_history(history, model_name)

        # Матрица ошибок
        print(f"\n[5/5] Построение матрицы ошибок...")
        cm_percent = self.plot_confusion_matrix_percent(model, test_generator, model_name)

        # Сохраняем историю
        with open(f'models/{model_name}_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        # Итоговые результаты
        best_epoch = np.argmax(history.history['val_accuracy'])
        best_accuracy = max(history.history['val_accuracy'])

        print(f"\n{'=' * 60}")
        print(f"ОБУЧЕНИЕ {model_name.upper()} ЗАВЕРШЕНО!")
        print(f"{'=' * 60}")
        print(f"Лучшая эпоха: {best_epoch + 1}")
        print(f"Лучшая точность на валидации: {best_accuracy:.3f}")
        print(f"Всего эпох обучено: {len(history.history['accuracy'])}")
        print(f"\nФайлы сохранены в папке 'models/':")
        print(f"  - {model_name}_best.keras")
        print(f"  - {model_name}_history.png")
        print(f"  - {model_name}_confusion_matrix.png")
        print(f"  - {model_name}_confusion_matrix.txt")

        return {
            'model': model,
            'history': history,
            'confusion_matrix': cm_percent,
            'name': model_name,
            'best_accuracy': best_accuracy,
            'best_epoch': best_epoch
        }


def main():
    # Укажите путь к вашему датасету
    DATASET_PATH = "C:/Users/Tamerlan/PycharmProjects/Human_Emotion_Detections_CV/dataset"

    # Создаем папку models если её нет
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✓ Создана папка models")

    # Проверяем наличие старых моделей
    existing_models = []
    if os.path.exists('vgg16_best.keras'):
        existing_models.append("VGG16")
    if os.path.exists('mobilenetv2_best.keras'):
        existing_models.append("MobileNetV2")

    if existing_models:
        print(f"\n✓ Найдены ранее обученные модели: {', '.join(existing_models)}")
        print("  Они будут сохранены и не будут перезаписаны")

    print("\n" + "=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ EFFICIENTNET")
    print("=" * 60)
    print("\nБудет обучена только одна модель:")
    print("  • EfficientNetV2-S (замена ResNet50)")
    print(f"  • Количество эпох: {EPOCHS}")
    print("  • Размер батча: 32")
    print("\nОстальные модели (VGG16, MobileNetV2) останутся без изменений")

    # Создаем тренер и обучаем
    trainer = EfficientNetTrainer(DATASET_PATH)
    result = trainer.train_model()

    if result:
        print("\n" + "=" * 60)
        print("ГОТОВО! ТЕПЕРЬ В MAIN.PY:")
        print("=" * 60)
        print("\nМодель EfficientNet автоматически загрузится в MAIN.py")
        print("Просто запустите:")
        print("  python MAIN.py")
        print("\nВ интерфейсе будет доступна модель 'EfficientNetV2-S'")


if __name__ == "__main__":
    main()