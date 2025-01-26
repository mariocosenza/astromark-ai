from flask import Flask, request

import gemini_service

app = Flask(__name__)


@app.route('/ai/orientation', methods=['POST'])
def gemini():
    data = request.get_json()  # Parse JSON from request body
    return gemini_service.get_llm(data)
@app.route('/ai/ticket', methods=['POST'])
def predict_tag():
    return request.form['message']


if __name__ == '__main__':
    app.run()
