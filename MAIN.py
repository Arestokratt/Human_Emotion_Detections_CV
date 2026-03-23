# MAIN.py
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
        self.current_model_name = None
        self.face_cascade = None

        # Разные размеры для разных моделей
        self.IMG_SIZE_224 = (224, 224)  # Для VGG16, MobileNet, EfficientNet
        self.IMG_SIZE_299 = (299, 299)  # Для Xception

        self.CLASS_NAMES = ['Злость', 'Радость', 'Грусть', 'Удивление']
        self.current_image = None
        self.current_image_path = None

        # Загружаем каскад Хаара и модели
        self.load_face_cascade()
        self.load_models_thread()

        # Создаем интерфейс
        self.create_widgets()

    def load_face_cascade(self):
        """Загрузка каскада Хаара"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_face(self, image):
        """Детекция лица"""
        if self.face_cascade is None:
            return True, None

        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_array = image

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return False, None

        (x, y, w, h) = faces[0]
        face_roi = img_array[y:y + h, x:x + w]
        return True, face_roi

    def load_models_thread(self):
        """Загружаем модели в отдельном потоке"""
        thread = threading.Thread(target=self.load_models)
        thread.daemon = True
        thread.start()

    def load_models(self):
        """Загрузка моделей"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'model')

        # Только нужные модели (БЕЗ ЗАПАСНЫХ)
        models_to_load = [
            ('xception_best.keras', 'Xception', self.IMG_SIZE_299),
            ('vgg16_best.keras', 'VGG16', self.IMG_SIZE_224),
            ('mobilenetv2_best.keras', 'MobileNetV2', self.IMG_SIZE_224)
        ]

        loaded_count = 0
        loaded_names = []

        for model_file, model_name, img_size in models_to_load:
            model_path = os.path.join(models_dir, model_file)
            try:
                if os.path.exists(model_path):
                    # Загружаем модель
                    model = tf.keras.models.load_model(model_path)

                    # Сохраняем модель и её размер
                    self.models[model_name] = {
                        'model': model,
                        'input_size': img_size
                    }

                    loaded_count += 1
                    loaded_names.append(model_name)
                    print(f"✓ Загружена: {model_name} (размер: {img_size[0]}x{img_size[1]})")
                else:
                    print(f"✗ Файл {model_file} не найден")
            except Exception as e:
                print(f"✗ Ошибка загрузки {model_name}: {e}")

        # Обновляем интерфейс
        if loaded_count > 0:
            # Выбираем первую модель по умолчанию
            first_model = list(self.models.keys())[0]
            self.current_model_name = first_model
            self.current_model = self.models[first_model]['model']

            status_text = f"Загружено моделей: {loaded_count} - {', '.join(loaded_names)}"
            self.root.after(0, self.update_status, status_text, "green")
            self.root.after(0, self.update_model_selector)
        else:
            self.root.after(0, self.update_status, "Модели не загружены ✗", "red")

    def update_status(self, text, color):
        """Обновление статуса"""
        self.status_label.config(text=text, foreground=color)

    def update_model_selector(self):
        """Обновление выпадающего списка"""
        if self.models:
            model_names = list(self.models.keys())
            self.model_combo['values'] = model_names
            self.model_combo.set(model_names[0])

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

        # Статус
        self.status_label = tk.Label(
            self.root,
            text="Загрузка моделей...",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='orange'
        )
        self.status_label.pack()

        # Выбор модели
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
            width=20
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

        # Изображение
        self.image_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.image_frame.pack(pady=10)
        self.image_label = tk.Label(self.image_frame, bg='#f0f0f0')
        self.image_label.pack()

        # Результаты
        self.result_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.result_frame.pack(pady=20)

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

        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 14, "bold"),
            bg='#f0f0f0',
            wraplength=500
        )
        self.result_label.pack(pady=5)

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
            text="Developed by: Galaov, Morozov, Shipul, Jdanov, Janabaev",
            font=("Arial", 9),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        footer.pack(side="bottom", pady=10)

    def on_model_change(self, event):
        """Смена модели"""
        model_name = self.model_combo.get()
        if model_name in self.models:
            self.current_model_name = model_name
            self.current_model = self.models[model_name]['model']
            print(f"✓ Выбрана модель: {model_name}")

    def upload_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )

        if file_path:
            self.current_image_path = file_path
            self.current_image = Image.open(file_path)

            display_image = self.current_image.copy()
            display_image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            self.analyze_btn.config(state="normal")
            self.result_label.config(text="")
            self.info_label.config(text="")

    def analyze_image(self):
        """Анализ изображения"""
        if not self.models:
            messagebox.showerror("Ошибка", "Модели не загружены!")
            return

        if self.current_image is None:
            return

        # Проверяем лицо
        has_face, face_roi = self.detect_face(self.current_image)

        if not has_face:
            self.result_label.config(
                text="Лицо не обнаружено!",
                foreground="red"
            )
            return

        # Запускаем анализ
        thread = threading.Thread(target=self.do_analysis, args=(face_roi,))
        thread.daemon = True
        thread.start()

        self.result_label.config(text="Анализ...", foreground="blue")
        self.analyze_btn.config(state="disabled")

    def do_analysis(self, face_roi):
        """Анализ изображения с голосованием"""
        try:
            from collections import Counter

            all_predictions = {}
            all_confidences = {}

            # Анализируем каждой моделью с её размером
            for model_name, model_info in self.models.items():
                model = model_info['model']
                input_size = model_info['input_size']

                # Изменяем размер под конкретную модель
                img_resized = cv2.resize(face_roi, input_size)
                img_array = img_resized.astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Предсказание
                preds = model.predict(img_array, verbose=0)[0]
                pred_index = np.argmax(preds)
                pred_label = self.CLASS_NAMES[pred_index]
                confidence = preds[pred_index]

                all_predictions[model_name] = pred_label
                all_confidences[model_name] = confidence

                print(f"{model_name}: {pred_label} ({confidence:.3f})")

            # ГОЛОСОВАНИЕ
            votes = Counter(all_predictions.values())
            final_emotion = votes.most_common(1)[0][0]

            # Средняя уверенность по всем моделям
            avg_confidence = np.mean(list(all_confidences.values()))

            # Формируем детальный отчет
            detail_text = "Результаты голосования:\n"
            for emotion, count in votes.most_common():
                detail_text += f"  {emotion}: {count} голос(а/ов)\n"

            detail_text += "\nРезультаты отдельных моделей:\n"
            for model_name in all_predictions:
                detail_text += f"  {model_name}: {all_predictions[model_name]} ({all_confidences[model_name] * 100:.1f}%)\n"

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
        """Обновление результата"""
        colors = {
            'Злость': '#d84315',
            'Радость': '#2e7d32',
            'Грусть': '#c0392b',
            'Удивление': '#1e88e5'
        }
        color = colors.get(label, 'black')

        self.result_label.config(
            text=f"Обнаружена эмоция: {label}\n"
                 f"Уверенность: {confidence * 100:.1f}%",
            foreground=color
        )

        self.info_label.config(
            text=detail_text,
            foreground="#555555"
        )

        self.analyze_btn.config(state="normal")


# Запуск
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()