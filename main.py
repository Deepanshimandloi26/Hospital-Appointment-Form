from flask import Flask, render_template, request, jsonify
from sklearn import base
from sklearn.model_selection import KFold
import mysql.connector
from mysql.connector import errorcode
import json
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib 
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])

@app.route('/store_entry', methods=['GET','POST'])
@cross_origin()
def store_text():
    # data = request.files['full_name']
    
    try:
        cnx = mysql.connector.connect(user='root',password='mysql',host='localhost',
                                        database='hospital')
        Con_obj = cnx.cursor()
        print(f"insert into appointment({','.join(str(val) for val in request.form.keys())},column_name) values({','.join('\''+str(val)+'\'' for val in request.form.values())},0)")

        Con_obj.execute(f"insert into appointment({','.join(str(val) for val in request.form.keys())},column_name) values({','.join('\''+str(val)+'\'' for val in request.form.values())},0)")
        cnx.commit()
        predict()
        return json.dumps(str(request.json))
    except mysql.connector.Error as err:
        return str(err)
needed_training_dataset = pd.read_csv(r"C:\\Users\\somya\\Downloads\\majorc\\Major Frontend\\needed_training_dataset.csv")
@app.route('/login',methods = ['POST'])
@cross_origin()
def login():

    cnx = mysql.connector.connect(user='root',password='mysql',host='localhost',
                                        database='hospital')
    Con_obj = cnx.cursor()
    Con_obj.execute(f"select * from users where username = '{request.form['uname']}' and password = '{request.form['psw']}'")
    data = Con_obj.fetchall()
    return jsonify(str(len(data)))
@app.route('/download',methods = ['GET'])
@cross_origin()
def download():
    cnx = mysql.connector.connect(user='root',password='mysql',host='localhost',
                                        database='hospital')
    Con_obj = cnx.cursor()
    Con_obj.execute("select id,cast(begintime as SIGNED INTEGER),state,workflow_status,appt_type,description,event_short,category,date_of_birth,no_show from appointment")
    data = Con_obj.fetchall()
    if(len(data) < 1):
        f = open(r"C:\\Users\\somya\\Downloads\\majorc\\Major Frontend\\test.txt",'w')
        f.write("No New Entry")
        f.close()
        return jsonify("C:\\Users\\somya\\Downloads\\majorc\\Major Frontend\\test.txt")
    to_process = []
    for row in data:
        predict = []
        for i in row:
            predict.append(i)
        to_process.append(predict)
    keys = ["id","begintime", "state", "workflow_status", "appt_type", "description", "event_short", "location_name", "Age", "No_Show_Flag"]
    result = [{keys[i]: value for i, value in enumerate(sublist)} for sublist in data]
    final = pd.DataFrame(result)
    final.to_csv(r"C:\\Users\\somya\\Downloads\\majorc\\Major Frontend\\test.csv",index = False)
    return jsonify("C:\\Users\\somya\\Downloads\\majorc\\Major Frontend\\test.csv")
@app.route('/predict', methods=['GET'])
@cross_origin()
def predict():
    cnx = mysql.connector.connect(user='root',password='mysql',host='localhost',
                                        database='hospital')
    Con_obj = cnx.cursor()
    Con_obj.execute("select id,cast(begintime as SIGNED INTEGER),state,workflow_status,appt_type,description,event_short,category,date_of_birth,0 from appointment where column_name = '0'")
    data = Con_obj.fetchall()
    if(len(data) < 1):
        return jsonify("No New Entry")
    to_process = []
    for row in data:
        predict = []
        for i in row:
            predict.append(i)
        to_process.append(predict)
    keys = ["id","begintime", "state", "workflow_status", "appt_type", "description", "event_short", "location_name", "Age", "No_Show_Flag"]
    result = [{keys[i]: value for i, value in enumerate(sublist)} for sublist in data]
    add = result
    count = 1
    print(add)
    all_id = []
    for r in result:
        all_id.append(r.pop("id"))
        needed_training_dataset.loc[len(needed_training_dataset)] = r
        count += 1
    WEIGHT = len(needed_training_dataset.index) / 2
    target1 = KFoldTargetEncoderTrain('workflow_status','No_Show_Flag',n_fold=2,weight = WEIGHT)
    target2 = KFoldTargetEncoderTrain('description','No_Show_Flag',n_fold=2,weight = WEIGHT)
    target3 = KFoldTargetEncoderTrain('location_name','No_Show_Flag',n_fold=2,weight = WEIGHT)
    target4 = KFoldTargetEncoderTrain('Age','No_Show_Flag',n_fold=2,weight = WEIGHT)
    target5 = KFoldTargetEncoderTrain('begintime','No_Show_Flag',n_fold=2,weight = WEIGHT)
    target6 = KFoldTargetEncoderTrain('appt_type','No_Show_Flag',n_fold=2,weight = WEIGHT)
    target7 = KFoldTargetEncoderTrain('event_short','No_Show_Flag',n_fold=2,weight = WEIGHT)
    target8 = KFoldTargetEncoderTrain('state','No_Show_Flag',n_fold=2,weight = WEIGHT)
    predict_data = needed_training_dataset.tail(count-1)
    predict_data = target1.fit_transform(predict_data)
    predict_data = target2.fit_transform(predict_data)
    predict_data = target3.fit_transform(predict_data)
    predict_data = target4.fit_transform(predict_data)
    predict_data = target5.fit_transform(predict_data)
    predict_data = target6.fit_transform(predict_data)
    predict_data = target7.fit_transform(predict_data)
    predict_data = target8.fit_transform(predict_data)
    encoded_testing_dataset = predict_data[['appt_type_enc','Age_enc','begintime_enc','state_enc','location_name_enc','workflow_status_enc','description_enc','event_short_enc','No_Show_Flag']]
    Y1 = encoded_testing_dataset['No_Show_Flag']
    X1 = encoded_testing_dataset.drop(['No_Show_Flag'],axis = 1)
    xgb_test = xgb.DMatrix(X1, label = Y1)
    xgb_model = joblib.load(r'C:\\Users\\somya\\Downloads\\majorc\\Major Frontend\\trained.joblib')
    xgb_predict = xgb_model.predict(xgb_test)
    xgb_predict = pd.DataFrame(xgb_predict, columns = ['No_Show'])
    xgb_predict = np.where(xgb_predict['No_Show'] >= 0.65,1,0)
    xgb_predict = pd.DataFrame(xgb_predict, columns = ['No_Show'])
    no_show = []
    for ind in xgb_predict.index:
        
        no_show.append(xgb_predict['No_Show'][ind])
    for i in range(len(no_show)):
        Con_obj = cnx.cursor()
        Con_obj.execute(f"update appointment set no_show = {no_show[i]} , column_name = 1  where id = {all_id[i]}")
        cnx.commit()
        print("commit")
    # final = pd.concat([xgb_predict,pd.DataFrame(predict_data.tail(count-1))],ignore_index=True)
    # predict_data.to_csv(r"C:\\Users\\somya\\Downloads\\majorc\\Major Frontend\\test.csv",index = False)
    return (all_id)

class KFoldTargetEncoderTrain(base.BaseEstimator,
                               base.TransformerMixin):
    def __init__(self,colnames,targetName,
                  n_fold=5, verbosity=True,
                  discardOriginal_col=False,
                  weight = 300000):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
        self.weight = weight
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        agg = needed_training_dataset.groupby(self.colnames)[self.targetName].agg(['count', 'mean'])
        mean = needed_training_dataset[self.targetName].mean()
        counts = agg['count']
        means = agg['mean']
        mean_of_target = (counts * means + self.weight * mean) / (counts + self.weight)
        kf = KFold(n_splits = self.n_fold)
        col_mean_name = self.colnames + '_' + 'enc'
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(mean_of_target)                                          
            X[col_mean_name].fillna(mean_of_target,inplace = True)
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName,                    
                   np.corrcoef(X[self.targetName].values,
                               encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X
if __name__ == '__main__':
    app.run(debug=True)



