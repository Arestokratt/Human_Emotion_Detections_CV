import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk
import os
import threading
from collections import Counter


class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Emotion Detection")
        self.root.geometry("1400x700")
        self.root.configure(bg='#f0f0f0')

        # Загружаем модели
        self.models = {}
        self.face_cascade = None

        # Разные размеры для разных моделей
        self.IMG_SIZE_224 = (224, 224)  # Для VGG16, MobileNet
        self.IMG_SIZE_299 = (299, 299)  # Для Xception

        self.CLASS_NAMES = ['Злость', 'Радость', 'Грусть', 'Удивление']
        self.current_image = None
        self.current_image_path = None

        # Цвета для эмоций
        self.emotion_colors = {
            'Злость': '#d84315',
            'Радость': '#2e7d32',
            'Грусть': '#c0392b',
            'Удивление': '#1e88e5'
        }

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

        # Модели для загрузки
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
                    print(f"✗ Файл {model_file} не найден в {models_dir}")
            except Exception as e:
                print(f"✗ Ошибка загрузки {model_name}: {e}")

        # Обновляем статус
        if loaded_count > 0:
            status_text = f"Загружено моделей: {loaded_count} - {', '.join(loaded_names)} ✓"
            self.root.after(0, self.update_status, status_text, "green")
        else:
            self.root.after(0, self.update_status, "Модели не загружены ✗", "red")

    def update_status(self, text, color):
        """Обновление статуса"""
        self.status_label.config(text=text, foreground=color)

    def create_widgets(self):
        # Главный контейнер
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)

        # ЛЕВАЯ ЧАСТЬ (изображение и результат)
        left_frame = tk.Frame(main_container, bg='#f0f0f0')
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 20))

        # Заголовок
        title = tk.Label(
            left_frame,
            text="Распознавание эмоций",
            font=("Arial", 24, "bold"),
            bg='#f0f0f0',
            fg='#1a2a44'
        )
        title.pack(pady=(0, 20))

        # Статус модели
        self.status_label = tk.Label(
            left_frame,
            text="Загрузка моделей...",
            font=("Arial", 10),
            bg='#f0f0f0',
            fg='orange'
        )
        self.status_label.pack(pady=(0, 20))

        # Кнопка загрузки
        upload_btn = tk.Button(
            left_frame,
            text="📁 Выбрать изображение",
            command=self.upload_image,
            font=("Arial", 12),
            bg='#1a2a44',
            fg='white',
            padx=20,
            pady=10,
            cursor="hand2",
            relief='flat'
        )
        upload_btn.pack(pady=(0, 20))

        # Фрейм для изображения
        self.image_frame = tk.Frame(left_frame, bg='white', relief='solid', bd=1)
        self.image_frame.pack(pady=10)

        # Метка для изображения
        self.image_label = tk.Label(self.image_frame, bg='white')
        self.image_label.pack(padx=10, pady=10)

        # Фрейм для результата
        result_frame = tk.Frame(left_frame, bg='#f0f0f0')
        result_frame.pack(pady=20, fill='x')

        # Метка для результата
        self.result_label = tk.Label(
            result_frame,
            text="",
            font=("Arial", 18, "bold"),
            bg='#f0f0f0',
            wraplength=400
        )
        self.result_label.pack()

        # Кнопка анализа
        self.analyze_btn = tk.Button(
            left_frame,
            text="🔍 Анализировать",
            command=self.analyze_image,
            font=("Arial", 14, "bold"),
            bg='#27ae60',
            fg='white',
            padx=30,
            pady=12,
            state="disabled",
            cursor="hand2",
            relief='flat'
        )
        self.analyze_btn.pack(pady=20)

        # ПРАВАЯ ЧАСТЬ (голосование)
        right_frame = tk.Frame(main_container, bg='#f0f0f0', width=450)
        right_frame.pack(side='right', fill='both', expand=False)
        right_frame.pack_propagate(False)

        # Заголовок голосования
        vote_title = tk.Label(
            right_frame,
            text="🗳️ Результаты голосования",
            font=("Arial", 16, "bold"),
            bg='#f0f0f0',
            fg='#1a2a44'
        )
        vote_title.pack(pady=(0, 20))

        # Фрейм для результатов голосования
        self.vote_frame = tk.Frame(right_frame, bg='#f0f0f0')
        self.vote_frame.pack(fill='both', expand=True)

        # Изначально показываем сообщение
        self.vote_info_label = tk.Label(
            self.vote_frame,
            text="Загрузите изображение\nи нажмите 'Анализировать'",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#7f8c8d',
            justify='center'
        )
        self.vote_info_label.pack(expand=True)

        # Footer
        footer = tk.Label(
            self.root,
            text="Developed by: Galaov, Morozov, Shipul, Jdanov, Janabaev | Ансамбль из 3 моделей",
            font=("Arial", 9),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        footer.pack(side="bottom", pady=10)

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
            self.clear_vote_display()

    def clear_vote_display(self):
        """Очистка отображения голосования"""
        for widget in self.vote_frame.winfo_children():
            widget.destroy()

        self.vote_info_label = tk.Label(
            self.vote_frame,
            text="Ожидание анализа...",
            font=("Arial", 12),
            bg='#f0f0f0',
            fg='#7f8c8d',
            justify='center'
        )
        self.vote_info_label.pack(expand=True)

    def update_vote_display(self, all_predictions, all_confidences, votes):
        """Обновление отображения голосования"""
        # Очищаем фрейм
        for widget in self.vote_frame.winfo_children():
            widget.destroy()

        # Заголовок
        header = tk.Label(
            self.vote_frame,
            text="📊 Результаты голосования",
            font=("Arial", 12, "bold"),
            bg='#f0f0f0',
            fg='#1a2a44'
        )
        header.pack(pady=(0, 10))

        # Отображаем голоса
        vote_text = "🎯 Голосование:\n"
        for emotion, count in votes.most_common():
            vote_text += f"  {emotion}: {count} голос(а/ов)\n"

        vote_label = tk.Label(
            self.vote_frame,
            text=vote_text,
            font=("Arial", 11),
            bg='#f0f0f0',
            fg='#2c3e50',
            justify='left'
        )
        vote_label.pack(pady=(0, 15))

        # Разделитель
        separator = tk.Frame(self.vote_frame, height=2, bg='#bdc3c7')
        separator.pack(fill='x', pady=10)

        # Результаты отдельных моделей
        models_title = tk.Label(
            self.vote_frame,
            text="🤖 Результаты моделей:",
            font=("Arial", 11, "bold"),
            bg='#f0f0f0',
            fg='#1a2a44'
        )
        models_title.pack(pady=(10, 10))

        # Для каждой модели создаем строку с прогресс-баром
        for model_name in self.models.keys():
            if model_name in all_predictions:
                emotion = all_predictions[model_name]
                confidence = all_confidences[model_name]
                color = self.emotion_colors.get(emotion, '#7f8c8d')

                # Фрейм для одной модели
                model_frame = tk.Frame(self.vote_frame, bg='#f0f0f0')
                model_frame.pack(fill='x', pady=5)

                # Название модели
                name_label = tk.Label(
                    model_frame,
                    text=f"{model_name}:",
                    font=("Arial", 10, "bold"),
                    bg='#f0f0f0',
                    width=12,
                    anchor='w'
                )
                name_label.pack(side='left')

                # Эмоция
                emotion_label = tk.Label(
                    model_frame,
                    text=emotion,
                    font=("Arial", 10),
                    bg='#f0f0f0',
                    fg=color,
                    width=10,
                    anchor='w'
                )
                emotion_label.pack(side='left')

                # Процент
                percent_label = tk.Label(
                    model_frame,
                    text=f"{confidence * 100:.1f}%",
                    font=("Arial", 9),
                    bg='#f0f0f0',
                    width=8,
                    anchor='w'
                )
                percent_label.pack(side='left')

                # Прогресс-бар
                progress = ttk.Progressbar(
                    model_frame,
                    length=150,
                    mode='determinate',
                    value=confidence * 100
                )
                progress.pack(side='left', padx=(5, 0))

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
                text="❌ Лицо не обнаружено!",
                foreground="red"
            )
            return

        # Запускаем анализ
        thread = threading.Thread(target=self.do_analysis, args=(face_roi,))
        thread.daemon = True
        thread.start()

        self.result_label.config(text="⏳ Анализ...", foreground="blue")
        self.analyze_btn.config(state="disabled")

    def do_analysis(self, face_roi):
        """Анализ изображения с голосованием"""
        try:
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

            # Обновляем интерфейс
            self.root.after(0, self.update_result,
                            final_emotion, avg_confidence,
                            all_predictions, all_confidences, votes)

        except Exception as e:
            self.root.after(0, lambda: self.result_label.config(
                text=f"❌ Ошибка: {str(e)}",
                foreground="red"
            ))
            self.root.after(0, lambda: self.analyze_btn.config(state="normal"))

    def update_result(self, emotion, confidence, all_predictions, all_confidences, votes):
        """Обновление результата"""
        color = self.emotion_colors.get(emotion, 'black')

        self.result_label.config(
            text=f"🎭 {emotion}\n📊 Уверенность: {confidence * 100:.1f}%",
            foreground=color
        )

        # Обновляем отображение голосования
        self.update_vote_display(all_predictions, all_confidences, votes)

        self.analyze_btn.config(state="normal")


# Запуск
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()