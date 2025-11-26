import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import json
import os
from datetime import datetime

# --- 1. Настройки и загрузка модели ---
MODEL_PATH = "meat_classifier.pth"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "predictions.json")

# Создаем директорию для результатов, если её нет
os.makedirs(RESULTS_DIR, exist_ok=True)

# Классы определяются порядком, в котором они были в train_dataset
CLASSES = ['defective', 'non_defective'] 
# Вы можете сделать их более понятными для пользователя:
USER_CLASSES = ['🥩 ИСПОРЧЕННЫЙ / НЕГОДНЫЙ ПРОДУКТ', '✅ СВЕЖИЙ / ГОДНЫЙ ПРОДУКТ']

def check_model_trained(filepath):
    """Проверяет, существует ли файл модели и можно ли его загрузить."""
    if not os.path.exists(filepath):
        return False, f"Файл модели '{filepath}' не найден. Модель не обучена."
    
    try:
        # Пытаемся загрузить файл, чтобы проверить его валидность
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
        # Проверяем наличие необходимых ключей
        required_keys = ['model_state_dict', 'classifier']
        if not all(key in checkpoint for key in required_keys):
            return False, "Файл модели поврежден или неполный. Модель не обучена."
        return True, "Модель обучена и готова к использованию."
    except Exception as e:
        return False, f"Ошибка при проверке модели: {e}. Модель не обучена."

@st.cache_resource # Кэшируем загрузку модели, чтобы она не загружалась при каждом действии
def load_checkpoint(filepath):
    """Загружает модель из файла контрольной точки (.pth)."""
    try:
        # Загружаем контрольную точку, используя CPU, чтобы избежать проблем с памятью GPU при инференсе
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
        
        # 1. Восстанавливаем архитектуру (ResNet-18)
        model = models.resnet18(weights=None)
        
        # 2. Восстанавливаем последний слой
        model.fc = checkpoint['classifier']
        
        # 3. Загружаем сохраненные веса
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval() # Переводим модель в режим инференса
        return model, checkpoint
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}. Убедитесь, что файл '{MODEL_PATH}' находится в этой же папке.")
        return None, None

# Проверяем, обучена ли модель
is_trained, training_status = check_model_trained(MODEL_PATH)

# Загружаем модель
if is_trained:
    model, checkpoint = load_checkpoint(MODEL_PATH)
else:
    model, checkpoint = None, None


# --- 2. Функция предсказания ---

# Трансформации для инференса (должны совпадать с valid_transforms из main.py)
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_and_predict(image_file, model, user_classes):
    """Предобработка изображения и получение предсказания."""
    if model is None:
        return None, None, None
        
    # Загрузка и трансформация изображения
    image = Image.open(image_file).convert("RGB")
    # Добавляем размерность батча
    tensor_image = test_transforms(image).unsqueeze(0) 

    # Получение предсказания
    with torch.no_grad():
        output = model(tensor_image)
        probabilities = torch.softmax(output, dim=1)
        
    # Выбираем класс с максимальной вероятностью
    top_p, top_class = probabilities.topk(1, dim=1)
    
    # Возвращаем имя класса
    predicted_class_index = top_class.item()
    confidence = top_p.item() * 100
    
    # Определяем, свежее ли мясо
    is_fresh = predicted_class_index == 1  # non_defective = 1
    
    return user_classes[predicted_class_index], confidence, is_fresh

def save_prediction_result(image_name, result_class, confidence, is_fresh):
    """Сохраняет результат предсказания в JSON файл."""
    result = {
        "timestamp": datetime.now().isoformat(),
        "image_name": image_name,
        "prediction": result_class,
        "confidence": round(confidence, 2),
        "is_fresh": is_fresh,
        "class_index": 1 if is_fresh else 0
    }
    
    # Загружаем существующие результаты
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    else:
        results = []
    
    # Добавляем новый результат
    results.append(result)
    
    # Сохраняем обратно
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return result

# --- 3. Основное приложение Streamlit ---

st.title("🥩 Проверка свежести мяса")
st.caption(f"Классификатор обучен на {len(CLASSES)} классах с помощью ResNet-18.")

# Показываем статус модели
if is_trained:
    st.success(f"✅ {training_status}")
else:
    st.error(f"❌ {training_status}")
    st.warning("⚠️ Для работы приложения необходимо обучить модель. Запустите `main.py` для обучения.")

if model is None:
    st.warning("⚠️ Приложение не может работать, пока модель не загружена.")
    st.info("💡 Если модель обучена, убедитесь, что файл `meat_classifier.pth` находится в корневой директории проекта.")
else:
    st.write("---")
    st.subheader("📤 Загрузка изображения")
    st.write("Загрузите изображение мяса, чтобы определить его свежесть.")
    
    uploaded_file = st.file_uploader(
        "Выберите файл изображения...", 
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Поддерживаемые форматы: JPG, JPEG, PNG, BMP, WEBP"
    )

    if uploaded_file is not None:
        # Отображение загруженного изображения
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption='Загруженное изображение', use_container_width=True)
        
        st.write("") 
        
        # Кнопка для запуска предсказания
        if st.button('🔍 Проверить свежесть мяса', type="primary", use_container_width=True):
            st.subheader("📊 Результат анализа:")
            
            # Сбрасываем указатель файла перед обработкой
            uploaded_file.seek(0)
            
            with st.spinner('🔄 Анализ изображения...'):
                result_class, confidence, is_fresh = preprocess_and_predict(uploaded_file, model, USER_CLASSES)
            
            if result_class is None:
                st.error("❌ Ошибка при обработке изображения.")
            else:
                # Сохраняем результат
                saved_result = save_prediction_result(
                    uploaded_file.name, 
                    result_class, 
                    confidence, 
                    is_fresh
                )
                
                # Вывод результата с акцентом на свежесть
                if is_fresh:
                    st.success(f"✅ **МЯСО СВЕЖЕЕ!**")
                    st.success(f"🎉 **Результат:** {result_class}")
                else:
                    st.error(f"⚠️ **МЯСО НЕСВЕЖЕЕ!**")
                    st.error(f"❌ **Результат:** {result_class}")
                    st.warning("🔴 **ВНИМАНИЕ:** Рекомендуется не употреблять этот продукт в пищу!")
                
                st.info(f"📈 **Уверенность модели:** {confidence:.2f}%")
                
                # Показываем информацию о сохранении
                st.success(f"💾 Результат сохранен в файл: `{RESULTS_FILE}`")
                
                # Показываем детали сохраненного результата
                with st.expander("📋 Детали результата"):
                    st.json(saved_result)
    
    # Показываем историю результатов
    if os.path.exists(RESULTS_FILE):
        st.write("---")
        st.subheader("📜 История проверок")
        
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            try:
                all_results = json.load(f)
                if all_results:
                    st.write(f"Всего проверок: **{len(all_results)}**")
                    
                    # Показываем последние 5 результатов
                    recent_results = all_results[-5:]
                    st.write("**Последние 5 проверок:**")
                    for i, res in enumerate(reversed(recent_results), 1):
                        status_icon = "✅" if res['is_fresh'] else "❌"
                        status_text = "Свежее" if res['is_fresh'] else "Несвежее"
                        timestamp = datetime.fromisoformat(res['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                        st.write(f"{i}. {status_icon} {res['image_name']} - {status_text} ({res['confidence']}%) - {timestamp}")
                    
                    if st.button("🗑️ Очистить историю"):
                        os.remove(RESULTS_FILE)
                        st.success("История очищена!")
                        st.rerun()
                else:
                    st.info("История проверок пуста.")
            except json.JSONDecodeError:
                st.warning("Ошибка при чтении истории результатов.")