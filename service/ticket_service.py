from .pipeline import ClassifierType
from .report_predict import predict_category

def ticket_service(title, message):
    full_text = f"{title} {message}".strip()
    # Call the predict_category function; it should return a list of (label, probability)
    predictions = predict_category(full_text, ClassifierType.SVM)
    print("Predictions:", predictions)
    return predictions
