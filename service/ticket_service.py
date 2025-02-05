from .pipeline import ClassifierType
from .report_predict import predict_category

def ticket_service(title, message):
    full_text = f"{title} {message}".strip()
    # Call the predict_category function; it should return a list of (label, probability) tuples
    predictions = predict_category(full_text, ClassifierType.SVM)
    if predictions:
        first_category = predictions[0][0]
        print("First category:", first_category)
        return first_category
    else:
        print("No prediction obtained.")
        return "Category"

