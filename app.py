from flask import Flask, request, jsonify
from service.ticket_service import ticket_service
from service.gemini_service import get_llm  # if you need to use gemini_service

app = Flask(__name__)

@app.route('/ai/orientation', methods=['POST'])
def gemini():
    # Parse JSON from request body
    data = request.get_json()
    # Call your gemini_service; adjust the return value as needed
    response = get_llm(data)
    return jsonify({'response': response})

@app.route('/ai/ticket', methods=['POST'])
def predict_tag():
    # Get title and message from form data (or use JSON if you prefer)
    title = request.form.get('title', '')
    message = request.form.get('message', '')
    full_text = f"{title} {message}".strip()
    if not full_text:
        return jsonify({"error": "No title or message provided"}), 400

    predictions = ticket_service(title, message)
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
