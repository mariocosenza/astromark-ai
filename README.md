# AstroMark AI
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![SpaCy](https://img.shields.io/badge/nlp-SpaCy-green.svg)](https://spacy.io/)
[![Machine Learning](https://img.shields.io/badge/ml-Naive%20Bayes%20%26%20SVM-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/github/license/yourusername/astromark-ai.svg)](https://github.com/maricosenza/astromark-ai/blob/main/LICENSE)

<div align="center">
  <img src="static/logo.svg" alt="AstroMark Logo" width="200" height="200">
</div>

**Astromark AI** is an advanced machine learning project specifically designed to classify tickets destined for the school secretary. Leveraging Python 3.12, SpaCy, and sophisticated machine learning classifiers, the application provides intelligent and accurate ticket categorization.

## Key Features

- **Specialized Ticket Classification**: Developed to categorize school secretary tickets with high precision
- **Multiple Machine Learning Classifiers**: 
  - Naive Bayes Classifier
  - Support Vector Machine (SVM)
- **SpaCy-Powered Natural Language Processing**: Advanced text analysis and feature extraction
- **Flask RESTful API**: Easy retrieval of ticket categories via POST request
- **Python 3.12 Compatibility**: Utilizing the latest Python features and performance improvements

## Technology Stack

- **Language**: Python 3.12
- **NLP Library**: SpaCy
- **Classifiers**: 
  - Naive Bayes
  - Support Vector Machine (SVM)
- **Web Framework**: Flask
- **Deployment**: Easily deployable RESTful service

## API Endpoint

### Ticket Category Retrieval

To retrieve the category for a given ticket, send a POST request to the Flask endpoint:

```json
{
  "title": "Your ticket title",
  "message": "First ticket message"
}
```

**Response**:
```json
{
  "prediction": "specific_ticket_category"
}
```

## Getting Started

1. Clone the repository
2. Install dependencies
3. Run the application: `python app.py`
4. Send ticket classification requests to the Flask endpoint

## Contributing

Contributions are welcome! Please read the contributing guidelines before getting started.

## Authors

- [@mariocosenza](https://github.com/mariocosenza)
- [@mariofasolino](https://github.com/MarioFas)
- [@gsacrestano](https://github.com/gsacrestano)

