from flask import Flask, request

import gemini_service

app = Flask(__name__)


@app.route('/ai/orientation')
def gemini():
    return gemini_service.get_llm(
        request.args.get('message'))
@app.route('/ai/ticket', methods=['POST'])
def predict_tag():
    return request.form['message']


if __name__ == '__main__':
    app.run()
