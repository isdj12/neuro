import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 1. Определяем трансформации для изображений
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Загружаем данные
data_dir = 'dataset'
train_dataset = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
valid_dataset = datasets.ImageFolder(data_dir + '/validation', transform=valid_transforms)

# --------------------------------------------------------------------------------
# 3. Инициализация
num_classes = len(train_dataset.classes)
learning_rate = 0.001

# a) Проверяем наличие GPU и запрещаем запуск на CPU
if not torch.cuda.is_available():
    raise SystemError("❌ CUDA не обнаружена! Установи драйверы NVIDIA и PyTorch с поддержкой GPU.")

device = torch.device("cuda:0")
torch.cuda.set_device(device)

print(f"✅ Используется видеокарта: {torch.cuda.get_device_name(0)}")
print(f"Доступная память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} ГБ")

# b) Создаем модель
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Заменяем последний слой под нашу задачу
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Перемещаем модель на GPU
model = model.to(device)

# c) Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --------------------------------------------------------------------------------
# 4. DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

epochs = 10

# --------------------------------------------------------------------------------
# 5. Цикл обучения
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Валидация
    model.eval()
    validation_loss = 0.0
    corrects = 0
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    epoch_val_loss = validation_loss / len(valid_loader.dataset)
    epoch_acc = corrects.double() / len(valid_loader.dataset)

    print(f"Эпоха {epoch+1}/{epochs}.. "
          f"Loss обучения: {epoch_loss:.4f}.. "
          f"Loss валидации: {epoch_val_loss:.4f}.. "
          f"Точность на валидации: {epoch_acc:.4f}")

print("✅ Обучение завершено!")


model.class_to_idx = train_dataset.class_to_idx

# 2. Создаем словарь с метаданными и состоянием модели
checkpoint = {
    'input_size': 3 * 224 * 224, # (3 канала * 224 * 224)
    'output_size': num_classes,
    'class_to_idx': model.class_to_idx,
    'model_state_dict': model.state_dict(),
    'classifier': model.fc
}

# 3. Сохраняем контрольную точку
torch.save(checkpoint, 'meat_classifier.pth')

print("\n✅ Модель успешно сохранена в файл: meat_classifier.pth")

# --------------------------------------------------------------------------------
# 6. Информация о данных
print(f"Классы: {train_dataset.classes}")
print(f"Количество обучающих изображений: {len(train_dataset)}")
print(f"Количество валидационных изображений: {len(valid_dataset)}")
