# main.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk
import os
import threading
from collections import defaultdict


class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Emotion Detection")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # Загружаем модели
        self.models = {}
        self.current_model = None
        self.face_cascade = None

        # Константы
        self.IMG_SIZE = (224, 224)
        self.CLASS_NAMES = ['Злость', 'Радость', 'Грусть', 'Удивление']
        self.current_image = None
        self.current_image_path = None

        # Загружаем каскад Хаара и модели
        self.load_face_cascade()
        self.load_models_thread()

        # Создаем интерфейс
        self.create_widgets()

    def load_face_cascade(self):
        """Загрузка каскада Хаара для детекции лиц"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            print("Ошибка загрузки каскада Хаара")
            self.face_cascade = None

    def detect_face(self, image):
        """Детекция лица на изображении"""
        if self.face_cascade is None:
            return True, None  # Если каскад не загружен, пропускаем проверку

        # Конвертируем PIL Image в OpenCV формат
        if isinstance(image, Image.Image):
            # Конвертируем в RGB если нужно
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Конвертируем в numpy array
            img_array = np.array(image)
            # Конвертируем BGR для OpenCV
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_array = image

        # Конвертируем в оттенки серого
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        # Детектируем лица
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return False, None

        # Возвращаем первое найденное лицо
        (x, y, w, h) = faces[0]
        face_roi = img_array[y:y + h, x:x + w]

        return True, face_roi

    def load_models_thread(self):
        """Загружаем модели в отдельном потоке"""
        thread = threading.Thread(target=self.load_models)
        thread.daemon = True
        thread.start()

    def load_models(self):
        """Загрузка всех обученных моделей"""
        models_to_load = [
            ('model/resnet50_best.keras', 'ResNet50'),
            ('model/vgg16_best.keras', 'VGG16'),
            ('model/mobilenetv2_best.keras', 'MobileNetV2')
        ]

        loaded_count = 0
        for model_file, model_name in models_to_load:
            try:
                if os.path.exists(model_file):
                    self.models[model_name] = tf.keras.models.load_model(model_file)
                    loaded_count += 1
                    print(f"Загружена модель: {model_name}")
                else:
                    print(f"Файл {model_file} не найден")
            except Exception as e:
                print(f"Ошибка загрузки {model_name}: {e}")

        if loaded_count > 0:
            # Выбираем первую модель по умолчанию
            self.current_model = list(self.models.values())[0]
            self.root.after(0, self.update_status,
                            f"Загружено моделей: {loaded_count}/3 ✓", "green")

            # Обновляем выпадающий список
            self.root.after(0, self.update_model_selector)
        else:
            self.root.after(0, self.update_status,
                            "Модели не загружены ✗", "red")

    def update_status(self, text, color):
        """Обновление статуса в GUI"""
        self.status_label.config(text=text, foreground=color)

    def update_model_selector(self):
        """Обновление выпадающего списка моделей"""
        self.model_combo['values'] = list(self.models.keys())
        if self.models:
            self.model_combo.set(list(self.models.keys())[0])

    def create_widgets(self):
        # Заголовок
        title = tk.Label(
            self.root,
            text="Распознавание эмоций",
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#1a2a44'
        )
        title.pack(pady=20)

        # Статус модели
        self.status_label = tk.Label(
            self.root,
            text="Загрузка моделей...",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='orange'
        )
        self.status_label.pack()

        # Фрейм для выбора модели
        model_frame = tk.Frame(self.root, bg='#f0f0f0')
        model_frame.pack(pady=10)

        tk.Label(
            model_frame,
            text="Выберите модель:",
            font=("Arial", 11),
            bg='#f0f0f0'
        ).pack(side='left', padx=5)

        self.model_combo = ttk.Combobox(
            model_frame,
            font=("Arial", 11),
            state="readonly",
            width=15
        )
        self.model_combo.pack(side='left', padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_change)

        # Кнопка загрузки
        upload_btn = tk.Button(
            self.root,
            text="Выбрать изображение",
            command=self.upload_image,
            font=("Arial", 12),
            bg='#1a2a44',
            fg='white',
            padx=20,
            pady=10,
            cursor="hand2"
        )
        upload_btn.pack(pady=20)

        # Фрейм для изображения
        self.image_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.image_frame.pack(pady=10)

        # Метка для изображения
        self.image_label = tk.Label(self.image_frame, bg='#f0f0f0')
        self.image_label.pack()

        # Фрейм для результата
        self.result_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.result_frame.pack(pady=20)

        # Кнопка анализа
        self.analyze_btn = tk.Button(
            self.result_frame,
            text="Анализировать",
            command=self.analyze_image,
            font=("Arial", 12),
            bg='#27ae60',
            fg='white',
            padx=20,
            pady=10,
            state="disabled",
            cursor="hand2"
        )
        self.analyze_btn.pack()

        # Фрейм для результатов всех моделей
        self.all_results_frame = tk.Frame(self.result_frame, bg='#f0f0f0')
        self.all_results_frame.pack(pady=10)

        # Метка для основного результата
        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            wraplength=500
        )
        self.result_label.pack(pady=5)

        # Метка для дополнительной информации
        self.info_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='#555555',
            wraplength=600,
            justify='left'
        )
        self.info_label.pack(pady=5)

        # Footer
        footer = tk.Label(
            self.root,
            text="Developed by: Tasneem Bin Mahmood",
            font=("Arial", 9),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        footer.pack(side="bottom", pady=10)

    def on_model_change(self, event):
        """Обработчик смены модели"""
        model_name = self.model_combo.get()
        if model_name in self.models:
            self.current_model = self.models[model_name]

    def upload_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )

        if file_path:
            self.current_image_path = file_path
            self.current_image = Image.open(file_path)

            # Отображаем изображение
            display_image = self.current_image.copy()
            display_image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Активируем кнопку анализа
            self.analyze_btn.config(state="normal")

            # Очищаем результаты
            self.result_label.config(text="")
            self.info_label.config(text="")

    def analyze_image(self):
        """Анализ изображения"""
        if not self.models:
            messagebox.showerror("Ошибка", "Модели не загружены!")
            return

        if self.current_image is None:
            return

        # Проверяем наличие лица
        has_face, face_roi = self.detect_face(self.current_image)

        if not has_face:
            self.result_label.config(
                text="Лицо не обнаружено!",
                foreground="red"
            )
            self.info_label.config(
                text="Пожалуйста, загрузите изображение с четким изображением лица",
                foreground="red"
            )
            return

        # Запускаем анализ в отдельном потоке
        thread = threading.Thread(target=self.do_analysis, args=(face_roi,))
        thread.daemon = True
        thread.start()

        # Показываем статус
        self.result_label.config(text="Анализ...", foreground="blue")
        self.analyze_btn.config(state="disabled")

    def do_analysis(self, face_roi):
        """Выполнение анализа всеми моделями"""
        try:
            # Подготовка изображения лица
            if face_roi is not None:
                # Используем выделенное лицо
                img_resized = cv2.resize(face_roi, self.IMG_SIZE)
                img_array = img_resized.astype(np.float32) / 255.0
            else:
                # Если лицо не выделено, используем все изображение
                if self.current_image.mode != 'RGB':
                    img = self.current_image.convert('RGB')
                else:
                    img = self.current_image.copy()

                img_array = np.array(img)
                img_array = cv2.resize(img_array, self.IMG_SIZE)
                img_array = img_array.astype(np.float32) / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            # Анализ всеми моделями
            all_predictions = {}
            all_confidences = {}

            for model_name, model in self.models.items():
                preds = model.predict(img_array, verbose=0)[0]
                pred_index = np.argmax(preds)
                pred_label = self.CLASS_NAMES[pred_index]
                confidence = preds[pred_index]

                all_predictions[model_name] = pred_label
                all_confidences[model_name] = confidence

            # Голосование между моделями
            votes = defaultdict(int)
            for pred in all_predictions.values():
                votes[pred] += 1

            final_emotion = max(votes, key=votes.get)
            avg_confidence = np.mean([all_confidences[m] for m in all_confidences])

            # Формируем детальный отчет
            detail_text = "Результаты всех моделей:\n"
            for model_name in all_predictions:
                detail_text += f"{model_name}: {all_predictions[model_name]} ({all_confidences[model_name] * 100:.1f}%)\n"

            # Обновляем интерфейс
            self.root.after(0, self.update_result,
                            final_emotion, avg_confidence, detail_text)

        except Exception as e:
            self.root.after(0, lambda: self.result_label.config(
                text=f"Ошибка: {str(e)}",
                foreground="red"
            ))
            self.root.after(0, lambda: self.analyze_btn.config(state="normal"))

    def update_result(self, label, confidence, detail_text):
        """Обновление результата в интерфейсе"""

        colors = {
            'Злость': '#d84315',
            'Радость': '#2e7d32',
            'Грусть': '#c0392b',
            'Удивление': '#1e88e5'
        }
        color = colors.get(label, 'black')

        # Основной результат
        self.result_label.config(
            text=f"Обнаруженная эмоция: {label}\n"
                 f"Средняя уверенность: {confidence * 100:.1f}%",
            foreground=color
        )

        # Детальная информация
        self.info_label.config(
            text=detail_text,
            foreground="#555555"
        )

        self.analyze_btn.config(state="normal")


# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()