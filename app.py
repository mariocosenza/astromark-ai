from flask import Flask, request, Response
from service.ticket_service import ticket_service
from service.gemini_service import get_llm

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app = Flask(__name__)

@app.route('/ai/orientation', methods=['POST'])
def gemini():
    # Parse JSON from request body (forcing JSON parsing)
    data = request.get_json(force=True)
    # Call your get_llm function (adjust this if necessary)
    response_text = get_llm(data)
    # Return a plain text response with UTF-8 encoding
    return Response(response_text, mimetype="text/plain; charset=utf-8")


@app.route('/ai/ticket', methods=['POST'])
def predict_tag():
    # Get title and message from form data (or JSON if preferred)
    title = request.form.get('title', '')
    message = request.form.get('message', '')
    full_text = f"{title} {message}".strip()
    if not full_text:
        return Response("No title or message provided", status=400, mimetype="text/plain; charset=utf-8")

    predictions = ticket_service(title, message)

    # Convert predictions to a plain text representation (for example, a simple string)
    response_text = f"Predictions: {predictions}"
    return Response(response_text, status=200, mimetype="text/plain; charset=utf-8")


if __name__ == '__main__':
    app.run(debug=True)
