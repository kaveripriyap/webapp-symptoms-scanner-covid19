import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import socket
from logisticregression import prepare_init_features


app = Flask(__name__, static_folder="templates", template_folder='templates')
model = pickle.load(open('model_logreg.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        x_list = request.form.getlist('scanner')
        X_user = prepare_init_features(x_list)
        prediction = model.predict(X_user)
        output = int(prediction[0])
        if output == 1.0:
            return render_template('index.html', covid_yes_no='You seem to have contracted COVID-19! Please consult a physician. Stay safe and you can also have a look at our resources or reach out to us.')
        else:
            return render_template('index.html', covid_yes_no='You do not seem to exhibit any preliminary COVID-19 symptoms. Stay safe and you can also have a look at our resources or reach out to us.')
        # print(request.form.getlist('scanner'))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

class PrefixMiddleware(object):

    def __init__(self, app, voc=True):
        self.app = app
        if voc:
            myip = self.get_myip()
            mytoken = os.getenv("VOC_PROXY_TOKEN")
            self.prefix = f'/hostip/{myip}:5000/vocproxy/{mytoken}'
        else:
            self.prefix = ''

    def __call__(self, environ, start_response):
        print(environ['PATH_INFO'], self.prefix)
        environ['SCRIPT_NAME'] = self.prefix
        return self.app(environ, start_response)

    def get_myip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 53))
        return s.getsockname()[0]

# set voc=False if you run on local computer
app.wsgi_app = PrefixMiddleware(app.wsgi_app, voc=False)

