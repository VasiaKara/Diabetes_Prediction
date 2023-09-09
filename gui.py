from flask import Flask, url_for, render_template, redirect, request, session
import sys
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model, model_from_json
import keras
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.framework import ops

graph = ops.get_default_graph()

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

app = Flask(__name__)
app.secret_key = "dim-vas"

decision_tree_entropy_pkl = open('MyModels/decision_tree_entropy_classifier.pkl', 'rb')
decision_tree_entropy_model = pickle.load(decision_tree_entropy_pkl)
decision_tree_gini_pkl = open('MyModels/decision_tree_gini_classifier.pkl', 'rb')
decision_tree_gini_model = pickle.load(decision_tree_gini_pkl)
naive_bayes_pkl = open('MyModels/naive_bayes_classifier.pkl', 'rb')
naive_bayes_model = pickle.load(naive_bayes_pkl)
svm_pkl = open('MyModels/svm_classifier.pkl', 'rb')
svm_model = pickle.load(svm_pkl)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=["POST", "GET"])
def form():
    if request.method == "GET":
        return render_template('form.html') 
    elif request.method == "POST":
        Age = request.form["Age"]
        session["Age"] = Age
        BMI = request.form["BMI"]
        session["BMI"] = BMI
        Education = request.form["Education"]
        session["Education"] = Education
        RelationshipStatus = request.form["RelationshipStatus"]
        session["RelationshipStatus"] = RelationshipStatus
        SleepHrsNight = request.form["SleepHrsNight"]
        session["SleepHrsNight"] = SleepHrsNight
        PhysActive = request.form["PhysActive"]
        session["PhysActive"] = PhysActive
        Income = request.form["Income"]
        session["Income"] = Income
        Insured = request.form["Insured"]
        session["Insured"] = Insured
        BPDia = request.form["BPDia"]
        session["BPDia"] = BPDia
        return redirect(url_for('model')) 

@app.route('/model', methods=["POST", "GET"])
def model():
    if request.method == "GET":
        return render_template('model.html')
    elif request.method == "POST":
        mdl = request.form["model"]
    return redirect(url_for('result', mdl = mdl))

@app.route('/<mdl>')
def result(mdl):
    if "Age" in session:
        if "BMI" in session:
            if "Education" in session:
                if "RelationshipStatus" in session:
                    if "SleepHrsNight" in session:
                        if "PhysActive" in session:
                            if "Income" in session:
                                if "Insured" in session:
                                    if "BPDia" in session:
                                            Age = session["Age"]
                                            BMI = session["BMI"]
                                            Education = session["Education"]
                                            RelationshipStatus = session["RelationshipStatus"]
                                            SleepHrsNight = session["SleepHrsNight"]
                                            PhysActive = session["PhysActive"]
                                            Income = session["Income"]
                                            Insured = session["Insured"]
                                            BPDia = session["BPDia"]
                                            HDLChol = 1.36
                                            case = np.array([[Age,BMI,Education,RelationshipStatus,SleepHrsNight,PhysActive,Income,Insured,BPDia,HDLChol]])
                                            case = pd.DataFrame(case)
                                          
      
    if(mdl == "Decision tree(entropy)"):
        clf = decision_tree_entropy_model 
    elif(mdl == "Decision tree(gini)"):
         clf = decision_tree_gini_model
    elif(mdl == "Naive Bayes"):
        clf = naive_bayes_model
    elif(mdl == "SVM"):
        clf = svm_model
    elif(mdl == "MLP"):
        global graph
        with graph.as_default():
            mlp_pkl = open('MyModels/mlp_classifier.pkl', 'rb')
            mlp_model = pickle.load(mlp_pkl)
            prediction = mlp_model.predict(case)
            prediction = (prediction > 0.5)
            if(prediction):
                prediction = 1
            else:
                prediction = 0 
    
    if(mdl != "MLP") :
        prediction = clf.predict(case) 
            
    if(prediction==1):
        output = "HIGH"
    else:
        output = "LOW"
    return render_template('result.html', output=output)

if __name__ == '__main__':
    app.run()