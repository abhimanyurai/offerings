# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:58:04 2021

@author: abhimanyu.rai
"""

import pandas as pd
import numpy as np
import seaborn as sns
from pycaret.classification import *
import pickle 


offering_dataset = pd.read_csv("./Files/Offering Data Pivoted.csv")



offering_dataset.drop(['6 City of Interview','8 Company', '10 Outlet Name', '43 - NFR Present or Not', '44 Area bracket (sq. feet)','Monthly Revenue (FR)', 'Rev per Sq. Ft (FR)', 'Margin per Sq. Ft (FR)',
       'Monthly Revenue (NFR)', 'NFR revenues as % of Total revenues',
       'Rev per Sq. ft (NFR)', 'Margin per Sq. Ft (NFR)', 'Monthly NFR Profit',
       'Cost as % of Revenue','Demographics - Male Population', 'Demographics - Female Population', 'Demographics - Total Literate Population','Latitude',
       'Longitude', 'Address'],axis=1,inplace = True)

offering_dataset.columns

offerings = ['Aggregated vehicle related requirements', 'Virtual C-store',
        'Truck Stops (Auto Repair / Rest)',
       'QSR / Restaurant / Dhaba',
       'One stop shop for daily farmer needs','Forecourt Advertising','Pharmacy']

trained_model = {}
trained_model_logs = {}
to_remove = []

for offering in offerings:
    
    to_remove = []
    to_remove = offerings.copy()
    to_remove.remove(offering)
    print(to_remove)
    
    dataset = offering_dataset.copy(deep=True)
    dataset.drop(to_remove, axis=1,inplace=True)
    
    data = dataset.sample(frac=0.9, random_state=786).reset_index(drop=True)
    data_unseen = dataset.drop(data.index).reset_index(drop=True)
    
    print('Data for Modeling: ' + str(data.shape))
    print('Unseen Data For Predictions: ' + str(data_unseen.shape))
    
    
    exp_mclf101 = setup(data = data, target = offering, session_id=123, html=False,log_experiment = True, experiment_name = 'NFR_Check', normalize = True, silent = True) 
       
    
    best = compare_models(include = ['nb','rf','ridge','et','dt'])
    trained_model_logs[offering]=pull()
    
    tuned_et = tune_model(best)
    
    final_et = finalize_model(tuned_et)
    
    print("Best Model for "+offering)
    print(final_et)
    
    save_model(final_et,'./Files/'+offering.split(" ")[0])
    trained_model[offering] = final_et
    
output = open('./Files/Trained_Models_v1', 'wb')
pickle.dump(trained_model, output)
output.close()




pd.to_pickle(trained_model_logs,'./Files/pickle_file.pkl')
    

