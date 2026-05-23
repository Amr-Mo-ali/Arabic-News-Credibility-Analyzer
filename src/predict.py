import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from sklearn.metrics import accuracy_score, f1_score


# First we download the trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("AmrMohamed21/arabert-fake-news")
model = AutoModelForSequenceClassification.from_pretrained("AmrMohamed21/arabert-fake-news")

# Now we build a function to clean the input text
def clean_text(text):
    """
    Clean the input text by removing special characters and extra spaces.

    Parameters:
    text (str): The input text to be cleaned.

    Returns:
    str: The cleaned text.
    """
    # Remove non_Arabic characters
    if not isinstance(text, str):  # ← هنا مباشرة
        return ""
    # Remove special characters and digits
    
    cleaned_text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)  # شيل التشكيل
    cleaned_text = re.sub(r'[أإآ]', 'ا', cleaned_text)  # normalize الهمزات
    # Remove punctuation   
    # Remove numbers
    cleaned_text = re.sub(r'\d+', '', cleaned_text)  # شيل الأرقام
    cleaned_text = re.sub(r'[^\u0621-\u064A\s]', '', cleaned_text)  # شيل علامات الترقيم   
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text




# Now we tokenize the input text and make a prediction
def predict(text):
    """
    Predict the label of the input text using the trained model.

    Parameters:
    text (str): The input text to be classified.

    Returns:
    dict: A dictionary containing the predicted label and confidence.
    """

    cleaned_text = clean_text(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    probabilities = torch.softmax(logits, dim=1)
    confidence = probabilities[0][predicted_label].item()
    return {
    "label": "fake" if predicted_label == 0 else "real",
    "confidence": round(confidence * 100, 2)
}

print(predict("""
في إطار الحرص على توفير أوجه الرعاية الكاملة لزوار بيت الله  استلم مشرفو بعثة حج الجمعيات الأهلية المخيمات الخاصة بحجاج الجمعيات فى مشعرى منى وعرفات، وذلك ضمن الاستعدادات النهائية لتصعيد الحجاج إلى مشعر عرفات يوم الاثنين المقبل لأداء الركن الحج الأعظم .

كما أطمئنت   الدكتورة مايا  يوميا وزيرة التضامن الاجتماعي هاتفيا على حجاج الجمعيات الأهلية ووجهت الوزيرة بتذليل أي عقبات تواجه الحجج مع المتابعة المستمرة  لتوفير الخدمات  خلال مناطق المشاعر فضلا عن المخيمات في منى وعرفات ،  كذلك الاستعداد لتصعيد   الحجاج الى عرفات  اعتبارا من يوم الاثنين المقبل كما يواصل الوعاظ والواعظات تقديم حلقات التوعية للحجاج والرد على كافة استفساراتهم كذلك توافر الأوتوبيسات الحديثة لنقل الحجاج إلى المشاعر المقدسة، كي يؤدوا المناسك في سهولة ويسر.

وتُنفذ وزارة التضامن الاجتماعي منظومة متكاملة للاطمئنان على حجاج الجمعيات الأهلية تتضمن تخصيص مشرف لكل 46 حاجاً، وإنشاء عيادات طبية في مقار الإقامة بالتنسيق مع وزارة الصحة، بالإضافة إلى التواصل الهاتفي المستمر من الدكتورة مايا مرسى وزيرة التضامن مع رئيس بعثة حج الجمعيات وأعضاء غرفة العمليات المركزية لحج الجمعيات بالأراضي المقدسة، وذلك للاطمئنان على الحجاج الذين وصلوا إلى مكة المكرمة والمدينة المنورة.
  """))