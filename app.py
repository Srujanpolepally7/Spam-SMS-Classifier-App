# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = r"C:\Users\admin\Desktop\Master's\Interview\Spam-SMS-Classifier-Deployment-master\spam-sms-mnb-model.pkl"
classifier = pickle.load(open(filename, 'rb'))
tranfer_pkl = r"C:\Users\admin\Desktop\Master's\Interview\Spam-SMS-Classifier-Deployment-master\cv-transform.pkl"
cv = pickle.load(open(tranfer_pkl,'rb'))
app = Flask(__name__, template_folder = r"C:\Users\admin\Desktop\Master's\Interview\Spam-SMS-Classifier-Deployment-master\templates")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)