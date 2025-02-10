import logging  # Standard Python logging module for logging messages

# Import necessary classes and functions from Flask for building the web application.
from flask import Flask, request, Response

# Import custom functions and classes from service modules.
from service.pipeline import select_default_classifier, ClassifierType  # For classifier selection and type handling
from service.ticket_service import ticket_service  # Service for processing ticket-related tasks
from service.gemini_service import get_llm  # Service for handling orientation requests via an LLM
from service.word2vec_pipeline import get_word2vec_model  # Function to load a word2vec model

# Initialize a Flask application instance.
app = Flask(__name__)

# Configure Flask to not force ASCII encoding in JSON responses (allows for Unicode characters).
app.config['JSON_AS_ASCII'] = False

# (Re)initialize Flask app. Note: Creating a new Flask instance here overrides the previous configuration,
# so ensure this is intentional or remove the duplicate if not needed.
app = Flask(__name__)

# Set up basic configuration for logging: set level to INFO and define the logging message format.
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# Define a route for handling POST requests to /ai/orientation.
@app.route('/ai/orientation', methods=['POST'])
def gemini():
    # Parse JSON data from the request body. The force=True parameter ensures that
    # even if the content type is not 'application/json', the data is parsed as JSON.
    data = request.get_json(force=True)

    # Process the JSON data using the get_llm function (likely interfacing with a language model).
    response_text = get_llm(data)

    # Return the processed response as plain text with UTF-8 encoding.
    return Response(response_text, mimetype="text/plain; charset=utf-8")


# Define a route for handling POST requests to /ai/ticket.
@app.route('/ai/ticket', methods=['POST'])
def predict_tag():
    # Retrieve 'title' and 'message' from the submitted form data.
    # If these fields are not provided, default to empty strings.
    title = request.form.get('title', '')
    message = request.form.get('message', '')

    # Concatenate title and message, and remove any extra whitespace.
    full_text = f"{title} {message}".strip()

    # If both title and message are missing, return a 400 Bad Request response.
    if not full_text:
        return Response("No title or message provided", status=400, mimetype="text/plain; charset=utf-8")

    # Process the ticket information using the ticket_service function.
    predictions = ticket_service(title, message)

    # Return the predictions as a plain text response with HTTP status 200 (OK).
    return Response(predictions, status=200, mimetype="text/plain; charset=utf-8")


# Uncomment the following line to load the word2vec model using an SVM classifier, if needed.
# get_word2vec_model(ClassifierType.SVM)

# Initialize the default classifier by calling select_default_classifier.
# This sets up the environment for classification tasks; if you wish to use the word2vec model instead,
# you may comment out this line.
select_default_classifier()

# Entry point of the application. When this module is run directly, start the Flask development server.
if __name__ == '__main__':
    app.run(debug=False)
