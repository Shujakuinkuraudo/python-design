import os
import time
from flask import Flask, request, render_template, send_from_directory
from eval import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

p = predict()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    core = request.form.get('core')
    if file:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpg'))
    time1 = time.time_ns()
    numbers = p(core)
    time2 = (time.time_ns() - time1) / 1e9
    return render_template('index.html', val1=time.time(), time=time2, numbers=numbers)


if __name__ == '__main__':
    app.run()