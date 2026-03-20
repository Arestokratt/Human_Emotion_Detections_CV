# train_xception.py
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
IMG_SIZE = (299, 299)  # Xception требует 299x299!
BATCH_SIZE = 32
EPOCHS = 30
CLASS_NAMES = ['Злость', 'Радость', 'Грусть', 'Удивление']


class XceptionTrainer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train_path = os.path.join(dataset_path, "train")
        self.test_path = os.path.join(dataset_path, "test")

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

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        try:
            train_generator = train_datagen.flow_from_directory(
                self.train_path,
                target_size=IMG_SIZE,  # 299x299 для Xception
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                shuffle=True,
                classes=list(self.class_mapping.keys())
            )

            test_generator = test_datagen.flow_from_directory(
                self.test_path,
                target_size=IMG_SIZE,  # 299x299 для Xception
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

    def create_xception_model(self):
        """Создание модели Xception"""
        print("\n" + "=" * 50)
        print("СОЗДАНИЕ МОДЕЛИ XCEPTION")
        print("=" * 50)

        try:
            base_model = applications.Xception(
                input_shape=(299, 299, 3),
                include_top=False,
                weights='imagenet'
            )
            print("✓ Загружен Xception")

            # Замораживаем базовую модель
            base_model.trainable = False
            print(f"✓ Всего слоев в базовой модели: {len(base_model.layers)}")

            # Создаем полную модель
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(len(CLASS_NAMES), activation='softmax')
            ], name='xception')

            # Компиляция
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            print("✓ Модель успешно создана")
            model.summary()

            return model, base_model

        except Exception as e:
            print(f"✗ Ошибка при создании модели: {e}")
            return None, None

    def plot_training_history(self, history, model_name):
        """Визуализация процесса обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'{model_name} - Точность модели')
        axes[0].set_xlabel('Эпохи')
        axes[0].set_ylabel('Точность')
        axes[0].legend()
        axes[0].grid(True)

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

        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=(12, 8))
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    vmin=0, vmax=100)

        plt.title(f'{model_name} - Матрица ошибок (%)', fontsize=16, pad=20)
        plt.ylabel('Истинные значения', fontsize=12)
        plt.xlabel('Предсказанные значения', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'models/{model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n{'=' * 50}")
        print(f"ОТЧЕТ ПО КЛАССАМ - {model_name}")
        print(f"{'=' * 50}")
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

        return cm_percent

    def train_model(self):
        """Обучение Xception модели"""

        print(f"\n{'=' * 60}")
        print("ОБУЧЕНИЕ МОДЕЛИ XCEPTION")
        print(f"{'=' * 60}")

        # Подготовка данных
        print("\n[1/5] Подготовка данных...")
        train_generator, test_generator = self.prepare_data()

        if train_generator is None or test_generator is None:
            print("Ошибка подготовки данных")
            return None

        # Создание модели
        print(f"\n[2/5] Создание модели...")
        model, base_model = self.create_xception_model()
        if model is None:
            return None

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'xception_best.keras',
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
        print(f"\n[3/5] Начало обучения...")
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
        if os.path.exists('xception_best.keras'):
            model = keras.models.load_model('xception_best.keras')
            print("✓ Загружена лучшая модель")

        # Визуализация
        print(f"\n[4/5] Построение графиков обучения...")
        self.plot_training_history(history, 'xception')

        # Матрица ошибок
        print(f"\n[5/5] Построение матрицы ошибок...")
        cm_percent = self.plot_confusion_matrix_percent(model, test_generator, 'xception')

        # Сохраняем историю
        with open('xception_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        best_accuracy = max(history.history['val_accuracy'])
        print(f"\n{'=' * 60}")
        print(f"ОБУЧЕНИЕ XCEPTION ЗАВЕРШЕНО!")
        print(f"{'=' * 60}")
        print(f"Лучшая точность на валидации: {best_accuracy:.3f}")

        return {
            'model': model,
            'history': history,
            'confusion_matrix': cm_percent,
            'name': 'xception',
            'best_accuracy': best_accuracy
        }


def main():
    DATASET_PATH = "C:/Users/Tamerlan/PycharmProjects/Human_Emotion_Detections_CV/dataset"

    if not os.path.exists('models'):
        os.makedirs('models')
        print("✓ Создана папка models")

    print("\n" + "=" * 60)
    print("ЗАПУСК ОБУЧЕНИЯ XCEPTION")
    print("=" * 60)
    print("\nБудет обучена модель Xception")
    print(f"• Размер входных изображений: 299x299")
    print(f"• Количество эпох: {EPOCHS}")

    trainer = XceptionTrainer(DATASET_PATH)
    result = trainer.train_model()

    if result:
        print("\n" + "=" * 60)
        print("ГОТОВО! ТЕПЕРЬ В MAIN.PY:")
        print("=" * 60)
        print("\nДобавьте в MAIN.py в список models_to_load:")
        print("  ('xception_best.keras', 'Xception')")


if __name__ == "__main__":
    main()