import joblib
import re
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- 1. Настройка приложения и моделей ---

# Создаем экземпляр FastAPI
app = FastAPI(title="Spam Detection API", version="1.0")

# Загружаем наши модели при старте приложения.

print("Загрузка модели-векторизатора...")
vectorizer = SentenceTransformer('spam_vectorizer_model') # Указываем путь к папке
print("Модель-векторизатор загружена.")

print("Загрузка модели-классификатора...")
classifier = joblib.load('spam_classifier.pkl')
print("Модель-классификатор загружена.")

# Определяем наш порог для модели
SPAM_THRESHOLD = 0.9

# --- 2. Определяем модель данных для входящего запроса ---

class Message(BaseModel):
    text: str

# --- 3. Копируем нашу функцию предобработки ---

def preprocess_for_bert(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'^(subject:|re:|fwd:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 4. Создаем эндпоинты (точки входа) API ---

@app.get("/")
def read_root():
    return {"message": "Spam Detection API is running. Отправьте POST-запрос на /predict"}

@app.post("/predict/")
def predict_spam(message: Message):
    """
    Принимает текст и возвращает предсказание (спам/не спам).
    """
    # 1. Предобработка текста
    processed_text = preprocess_for_bert(message.text)
    
    # 2. Получение эмбеддинга
    embedding = vectorizer.encode([processed_text])
    
    # 3. Получение вероятности спама от классификатора
    # predict_proba возвращает вероятности для классов [0, 1]
    spam_probability = classifier.predict_proba(embedding)[0][1]
    
    # 4. Применение порога для принятия решения
    is_spam = spam_probability >= SPAM_THRESHOLD
    
    # 5. Возвращаем результат в формате JSON
    return {
        "text": message.text,
        "is_spam": bool(is_spam),
        "spam_probability": float(spam_probability)
    }