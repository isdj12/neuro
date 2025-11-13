import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --- 1. Настройки и загрузка модели ---
MODEL_PATH = "meat_classifier.pth"
# Классы определяются порядком, в котором они были в train_dataset
CLASSES = ['defective', 'non_defective'] 
# Вы можете сделать их более понятными для пользователя:
USER_CLASSES = ['🥩 ИСПОРЧЕННЫЙ / НЕГОДНЫЙ ПРОДУКТ', '✅ СВЕЖИЙ / ГОДНЫЙ ПРОДУКТ']

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
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}. Убедитесь, что файл '{MODEL_PATH}' находится в этой же папке.")
        return None

# Загружаем модель
model = load_checkpoint(MODEL_PATH)


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
        return "Модель неактивна."
        
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
    
    return user_classes[predicted_class_index], confidence

# --- 3. Основное приложение Streamlit ---

st.title("🥩 Проверка качества продукта (Defective/Non-Defective)")
st.caption(f"Классификатор обучен на {len(CLASSES)} классах с помощью ResNet-18.")
st.write("Загрузите изображение продукта, чтобы нейросеть определила его свежесть/качество.")

if model is None:
    st.warning("Приложение не может работать, пока модель не загружена. Проверьте консоль.")
else:
    uploaded_file = st.file_uploader("Выберите файл изображения...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Отображение загруженного изображения
        st.image(uploaded_file, caption='Загруженное изображение', use_column_width=True)
        st.write("") 
        
        # Кнопка для запуска предсказания
        if st.button('Проверить качество'):
            st.subheader("Результат анализа:")
            
            with st.spinner('Анализ изображения...'):
                result_class, confidence = preprocess_and_predict(uploaded_file, model, USER_CLASSES)
            
            # Вывод результата
            if "ИСПОРЧЕННЫЙ" in result_class:
                st.error(f"⚠️ **КЛАССИФИКАЦИЯ:** {result_class}")
            else:
                st.success(f"🎉 **КЛАССИФИКАЦИЯ:** {result_class}")
                
            st.info(f"Уверенность модели: **{confidence:.2f}%**")