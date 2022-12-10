# save this as app.py
from flask import Flask, request,render_template
from markupsafe import escape
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
models = load_model("models/1")
class_name = ['NORMAL','PNEUMONIA']


scaler = StandardScaler()
model = pickle.load(open("model_pickle.pkl", 'rb'))


app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/bmi', methods=['GET','POST'])
def bmi():
    Body = ''
    beemi = ''
    if request.method == "POST":
        weight = float(request.form.get('berat'))
        height = float(request.form.get('tinggi'))
        beemi = round((weight / ((height / 100 ) ** 2)), 2)
        Body = round((weight / ((height / 100 ) ** 2)), 2)
        if(Body <= 18.5):
            Body = 'Berat Badan Kurang'
        elif(Body <= 22.9):
            Body = 'Berat Badan Normal'
        elif(Body <= 29.9):
            Body = 'Berat Badan Berlebih (Kecenderungan Obesitas)'
        else:
            Body = 'Obesitas'
        return render_template("bmi.html", bmi_text="Hasil Body Mass Index adalah {}".format(Body), Bmi =beemi)
    else:        
        return render_template("bmi.html")

@app.route('/xray', methods=['GET','POST'])
def xray():
    if request.method == 'POST':
        f = request.files['xray']
        filename = f.filename
        target = os.path.join(APP_ROOT,'image/')
        des = "/".join([target,filename])
        f.save(des)
        
        test_image = image.load_img('image\\'+filename,target_size=(150,150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        prediction = models.predict(test_image)
        
        predicted_class = class_name[np.argmax(prediction[0])]
        confidence = round(np.max(prediction[0])*100)
        return render_template("xray.html", confidence= "Prediksi Terkena Pneumonia -> "+str(confidence)+ "%", prediction = "Hasil Prediksi X-Ray scan -> "+str(predicted_class))

    else:
        return render_template("xray.html")
    
@app.route('/predictive', methods=['GET','POST'])
def predictive():
    if request.method == "POST":
        gender = request.form['gender']
        age = request.form['age']
        hypertension = int(request.form['hypertension'])
        disease = int(request.form['disease'])
        married = request.form['married']
        work = request.form['work']
        residence = request.form['residence']
        glucose = float(request.form['glucose'])
        bmi = float(request.form['bmi'])
        smoking = request.form['smoking']
        
        #gender
        if (gender == "Male"):
            gender_male=1
            gender_other=0
        elif(gender == "Other"):
            gender_male=0
            gender_other=1
        else:
            gender_male=0
            gender_other=0
            
        #married
        if(married=="Yes"):
            married_yes =1
        else:
            married_yes=0
            
        #work Type
        if(work=='Self-employed'):
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 1
            work_type_children=0
        elif(work=='Private'):
            work_type_Never_worked = 0
            work_type_Private = 1
            work_type_Self_employed = 0
            work_type_children=0
        elif(work=='children'):
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children=1
        elif(work=='Never_worked'):
            work_type_Never_worked = 1
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children=0
        else:
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children=0
        
        #residence type
        if(residence=="Urban"):
            Residence_type_Urban=1
        else:
            Residence_type_Urban=0
            
        #Smokking 
        if(smoking =='formerly_smoked'):
            smoking_status_formerly_smoked = 1
            smoking_status_never_smoked = 0
            smoking_status_smokes = 0
        elif(smoking=='smokes'):
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_smokes = 1
        elif(smoking=='never_smoked'):
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 1
            smoking_status_smokes = 0
        else:
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_smokes = 0
        
        feature = scaler.fit_transform([[age, hypertension, disease, glucose, 
                                         bmi, gender_male, gender_other, married_yes, work_type_Never_worked, 
                                         work_type_Private, work_type_Self_employed, work_type_children, 
                                         Residence_type_Urban,smoking_status_formerly_smoked, smoking_status_never_smoked, 
                                         smoking_status_smokes]])
        
        prediction = model.predict(feature)[0]
        
        if prediction==0:
            prediction = "Kecil Resiko Terkena Serangan Stroke Tetap Jaga Kesehatan dan selalu hidup bersih.\n" 
        else:
            prediction = "Kecil Resiko Terkena Serangan Stroke Tetap Jaga Kesehatan dan selalu hidup bersih.\n" 

        return render_template("stroke-prediction.html", prediction_text="Prediksi Terkena Serangan Stroke --> {}".format(prediction))   
        
    else:
        return render_template("stroke-prediction.html")


if __name__ == "__main__":
    app.run(debug=True)