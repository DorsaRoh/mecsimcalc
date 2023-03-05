import pandas
import pandas as pd
import numpy as np
import numpy
import scipy.stats as st
import sklearn
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
import sklearn.metrics
from numpy import ravel
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import tree
import sys
import warnings
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn import model_selection
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sys import argv
import time
startTime = time.time()
import base64
import io
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from scipy.integrate import odeint
from matplotlib import animation
from matplotlib.colors import cnames
from matplotlib import cm
from scipy import integrate
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib as mpl
import os

def plt_show(plt, width, dpi=100):
    bytes = io.BytesIO()
    plt.savefig(bytes, format='png', dpi=dpi)
    if hasattr(plt, "close"):
        plt.close()
    bytes.seek(0)
    base64_string = "data:image/png;base64," + \
        base64.b64encode(bytes.getvalue()).decode("utf-8")
    return "<img src='" + base64_string + "' width='" + str(width) + "'>"
    
def main(inputs):
    
    if(inputs['disease']=='Brain Cancer Detection — Human + Mammal'):
        df = pd.read_csv("https://github.com/mvideet/MecSimCalc/blob/main/lungcancer.csv?raw=true")
        df = df.drop_duplicates()
        symptoms2 =  ['ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ', 'ALCOHOL CONSUMING','COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        symptoms =  ['GENDER', 'SMOKING','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ', 'ALCOHOL CONSUMING','COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN','LUNG_CANCER']
        df = df[symptoms]
        input_arr = [inputs['gen'],inputs['smok'],inputs['act'],inputs['fam'],inputs['injury'],inputs['seizures'],inputs['env'],inputs['obe'],inputs['exp'],inputs['genetic'],inputs['rad']]
        inp = pd.DataFrame([input_arr], columns = ['GENDER','SMOKING'] + symptoms2)
        df = df.replace({"M":0, "F":1, "NO":0, "YES":1})
        inp = inp.replace({"Male":0, "Female":1, "No":0, "Yes":1})
        print(inp)
        y = df["LUNG_CANCER"]
        X = df.drop(["LUNG_CANCER"],axis =1)
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=42)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)
        model = RandomForestClassifier()
        model.fit(X_train_scaled,y_train)
        y_pred = model.predict(inp)
        print(y_pred)
        test_pred = model.predict(X_test_scaled)
        print(metrics.accuracy_score(y_test,test_pred))
        if y_pred ==1:
            co5_l = """
            You most likely have lung and bronchial cancer. 

            Seek medical intervention immediately."""
            co6_l = "According to the American Cancer Society, lung cancer is the number one killer, with 2.2 million new cases in 2020 alone, making it three times deadlier than breast cancer: the second most deadliest cancer."
            co7_l = "Those diagnosed with lung cancer can experience shortness of breath, if the cancer grows to block the major airways; thus causing it to be a significantly dangerous maladie. Lung cancer additionally causes fluid to accumulate around the lungs, making it harder for the affected lung to expand fully when an individual inhales."
            co8_l = "Identifying those at highest risk of lung cancer, diagnosing as early as possible, and ensuring patients receive appropriate treatment at the correct time can prevent premature and consequential deaths. Access to noncommunicable disease medicines and basic health technologies is essential to ensure that those in need receive appropriate care."
            return{
                "co5_l":co5_l,
                "co6_l":co6_l,
                "co7_l":co7_l,
                "co8_l":co8_l
            }
        else:
            co1_l = """
        You most likely DO NOT have lung and/or bronchial cancer. If you are feeling similar symptoms, please seek medical intervention."""
            co2_l = "Lung cancer is the 2nd most common cancer worldwide. It is the most common cancer in men, and the 2nd most common cancer in women."
            co3_l = "If concerned, there are several ways you can reduce your risk of developing lung cancer, such as:"
            co4_l = """
    1. Avoiding cigarettes, smoking, second-hand smoke, and tobacco.
    2. Testing your home for the presence of radon.
    3. Avoiding carcinogens at work.
    4. Maintaining a healthy diet composed of fruits and vegetables.
    5. Consistently remain active for the majority of the week."""
            return{
                "co1_l":co1_l,
                "co2_l": co2_l,
                "co3_l": co3_l,
                "co4_l": co4_l
            }

    elif(inputs['disease']=='Brain Cancer Detection — Reptile + Ave/Bird'):
        df = pd.read_csv("https://github.com/mvideet/MecSimCalc/blob/main/lungcancer.csv?raw=true")
        df = df.drop_duplicates()
        symptoms2 =  ['ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ', 'ALCOHOL CONSUMING','COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        symptoms =  ['GENDER', 'SMOKING','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ', 'ALCOHOL CONSUMING','COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN','LUNG_CANCER']
        df = df[symptoms]
        input_arr = [inputs['gen'],inputs['smok'],inputs['act'],inputs['fam'],inputs['injury'],inputs['seizures'],inputs['env'],inputs['obe'],inputs['exp'],inputs['genetic'],inputs['rad']]
        inp = pd.DataFrame([input_arr], columns = ['GENDER','SMOKING'] + symptoms2)
        df = df.replace({"M":0, "F":1, "NO":0, "YES":1})
        inp = inp.replace({"Male":0, "Female":1, "No":0, "Yes":1})
        print(inp)
        y = df["LUNG_CANCER"]
        X = df.drop(["LUNG_CANCER"],axis =1)
        from sklearn.model_selection import train_test_split
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=42)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)
        model = RandomForestClassifier()
        model.fit(X_train_scaled,y_train)
        y_pred = model.predict(inp)
        print(y_pred)
        test_pred = model.predict(X_test_scaled)
        print(metrics.accuracy_score(y_test,test_pred))
        if y_pred ==1:
            co5_l = """
            You most likely have lung and bronchial cancer. 

            Seek medical intervention immediately."""
            co6_l = "According to the American Cancer Society, lung cancer is the number one killer, with 2.2 million new cases in 2020 alone, making it three times deadlier than breast cancer: the second most deadliest cancer."
            co7_l = "Those diagnosed with lung cancer can experience shortness of breath, if the cancer grows to block the major airways; thus causing it to be a significantly dangerous maladie. Lung cancer additionally causes fluid to accumulate around the lungs, making it harder for the affected lung to expand fully when an individual inhales."
            co8_l = "Identifying those at highest risk of lung cancer, diagnosing as early as possible, and ensuring patients receive appropriate treatment at the correct time can prevent premature and consequential deaths. Access to noncommunicable disease medicines and basic health technologies is essential to ensure that those in need receive appropriate care."
            return{
                "co5_l":co5_l,
                "co6_l":co6_l,
                "co7_l":co7_l,
                "co8_l":co8_l
            }
        else:
            co1_l = """
        You most likely DO NOT have lung and/or bronchial cancer. If you are feeling similar symptoms, please seek medical intervention."""
            co2_l = "Lung cancer is the 2nd most common cancer worldwide. It is the most common cancer in men, and the 2nd most common cancer in women."
            co3_l = "If concerned, there are several ways you can reduce your risk of developing lung cancer, such as:"
            co4_l = """
    1. Avoiding cigarettes, smoking, second-hand smoke, and tobacco.
    2. Testing your home for the presence of radon.
    3. Avoiding carcinogens at work.
    4. Maintaining a healthy diet composed of fruits and vegetables.
    5. Consistently remain active for the majority of the week."""
            return{
                "co1_l":co1_l,
                "co2_l": co2_l,
                "co3_l": co3_l,
                "co4_l": co4_l
            }
