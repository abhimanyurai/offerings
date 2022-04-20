# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:55:31 2021

@author: abhimanyu.rai
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from pycaret.classification import *
import streamlit as st
import json
import logging
import pickle
import googlemaps
from IPython.display import HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots




st.set_page_config(
    page_title="IOCL NFR Offerings Selector", layout="wide", page_icon="./Files/IOCL.png"
)

def add_element(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)



def main(input_df,training_df,input_df_cleaned_for_prediction,offerings,trained_models,log,input_distance_dict,API_key,financials_df,financials_df_cost_heads):
    
    st.sidebar.empty()
    cola,colb = st.columns([5,1])
    with cola:
        st.title('Welcome to NFR Offerings Simulator')
        st.subheader('Powered by Accenture')
    with colb:
        img = Image.open("./Files/IOCL.png")
   
        st.image(img, width=70)   

    st.markdown('---')
    
    pred,coordinates = Offerings_Input(input_df,training_df,input_df_cleaned_for_prediction,offerings)
    #st.write(pred)
    
    arr_key = []
    arr_val = []
    arr_key = list(pred.keys())
    
    for value in pred.values():
        arr_val.append(value[0])
  
    model_input = pd.DataFrame(data =np.array(arr_val).reshape(1,-1),columns=np.array(arr_key))
    
    
    score_list = {}
    
    for key in trained_models.keys():
        
        result = predict_model(trained_models[key],model_input, raw_score =True)
        
        
        scores = result[result.columns[pd.Series(result.columns).str.startswith('Score')]].T
        
        if scores.empty:
            score_list[key] = 0
        else:
            score_list[key]=scores[0].iloc[1]
      
    scores = pd.DataFrame(list(score_list.items()),columns=["attribute","value"])
  
    
    scores.sort_values(by="value",inplace = True, ascending=False)
    #scores.reset_index(inplace=True)
    #scores['attribute'] = scores['attribute'].apply(lambda x: x.split("_")[1])
            
    
    #st.header('Top Predicted Offerings')
    st.markdown(f"<h1 style='text-align: left; color: black;'>Top Predicted Offerings</h1>", unsafe_allow_html=True)
    col1,col2,col3,col4,col5 = st.columns([1,0.1,1,0.1,1])
    with col1:
        val = round(scores.iloc[0][1]*100,1)
        if val>=99:
            val=round(np.random.uniform(99.0,99.9),1)            
        
        if val>=20:
            
            text = scores.iloc[0][0]
            if text == "Virtual C-store":
                text = "Phygital Grocery"
            st.markdown(f"<h3 style='text-align: center; height: 50px; color: black ;'>{text}</h3>", unsafe_allow_html=True)        
            value = str(val)+"%"
            
            if val <= 50:
                st.markdown(f"<h1 style='text-align: center; background-color: #fb843b; color: white;'>{value}</h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 style='text-align: center; background-color: #000032; color: white;'>{value}</h1>", unsafe_allow_html=True)
    with col2:
        st.write("")
    with col3:
        val = round(scores.iloc[1][1]*100,1)
        if val>=99:
            val=round(np.random.uniform(99.0,99.9),1)
            
        if val>=20:
            text = scores.iloc[1][0]
            if text == "Virtual C-store":
                text = "Phygital Grocery"
            st.markdown(f"<h3 style='text-align: center; color:black; height: 50px ;'>{text}</h3>", unsafe_allow_html=True)
            value = str(val)+"%"
            
            if val <= 50:
                st.markdown(f"<h1 style='text-align: center; background-color: #fb843b; color: white;'>{value}</h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 style='text-align: center; background-color: #000032; color: white;'>{value}</h1>", unsafe_allow_html=True)
    with col4:
        st.write("")
    with col5:
        val = round(scores.iloc[2][1]*100,1)
        if val>=99:
            val=round(np.random.uniform(99.0,99.9),1)   
            
        if val>=20:
            text = scores.iloc[2][0]
            if text == "Virtual C-store":
                text = "Phygital Grocery"
            st.markdown(f"<h3 style='text-align: center; color: black ; height:50px; '>{text}</h3>", unsafe_allow_html=True)
            value = str(round(scores.iloc[2][1]*100,1))+"%"
            
            
            if val <= 50:
                st.markdown(f"<h1 style='text-align: center; background-color: #fb843b; color: white;'>{value}</h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h1 style='text-align: center; background-color: #000032; color: white;'>{value}</h1>", unsafe_allow_html=True)
    
    st.write("Legend")
    col7,col8,col9 = st.columns([1,1,8])
    with col7:
        st.markdown(f"<h6 style='text-align: center; color:white;background-color: #000032; height: 20px; width: 90px ; padding-right: 1px; '>Sweet Spot</h6>", unsafe_allow_html=True)
    with col8:
        st.markdown(f"<h6 style='text-align: center; color:white;background-color: #fb843b; height: 20px; width: 90px ;'>Hot Spot</h6>", unsafe_allow_html=True)
    with col9:
        st.write("")
            
    
    
    #NEARBY STORE SELECTORS
    st.markdown('---')
    st.markdown(f"<h1 style='text-align: left; color: black;'>Offerings at nearby stores</h1>", unsafe_allow_html=True)
    #st.header('Offerings at nearby stores')
    df_to_print = pd.DataFrame()
    if coordinates==-1:
        source_add = st.text_input('Enter Store Address')
        if source_add =="":
            print("Please enter address")
        else:
            #source_add = "Gurgaon"
            df_to_print = distance_calculator_custom(source_add, training_df, API_key, offerings)[0]
            st.write(HTML(df_to_print.to_html(index=False,justify='center')))
    else:
                
        df_to_print = input_distance_dict[coordinates]
        st.write(HTML(df_to_print.to_html(index=False, justify = 'center')))
        
        #st.write(distance_df[['6 City of Interview','8 Company','10 Outlet Name','Address','Distance (in KM)']].head())
      
        
    #FINANCIALS DISPLAY
    st.markdown('---')
    st.markdown(f"<h1 style='text-align: left; color: black;'>RO Financials</h1>", unsafe_allow_html=True)
    #st.header('RO Financials')
    
    fin_list = list(financials_df['Idea'].drop_duplicates())
    print(fin_list)
    
    offer_financial = st.selectbox('Offering', fin_list)
    
    #offer_financial = 'Aggregated vehicle related requirements'
    print(offer_financial)
    
    financials_df_sliced = financials_df[financials_df['Idea']==offer_financial][['Value','FY25','FY30','FY35','FY40']]
    
    if financials_df_sliced.empty:
        st.write("Financials Data Unavailable")
    else:
        data_x = ['FY25','FY30','FY35','FY40']
        data_y1 = financials_df_sliced[financials_df_sliced['Value']=='Revenue'][data_x].iloc[0]
        data_y2 = financials_df_sliced[financials_df_sliced['Value']=='Profitability'][data_x].iloc[0]
        data_y1_new=[]
        for y in data_y1:
            data_y1_new.append(float(y))
        data_y2_new=[]
        for y in data_y2:
            data_y2_new.append(float(y.split("%")[0])/100)
       
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        fig.add_trace(
            go.Bar(x=data_x, y=data_y1_new, name="Revenue Data", marker_color='#000032'),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Line(x=data_x, y=data_y2_new, name="Profitability Data"),
            secondary_y=True,
        )
        
       

        # Set x-axis title
        fig.update_xaxes(title_text="Year")

        
        fig.update_layout(width=1000,height=400)

        # Set y-axes titles
        fig.update_yaxes(title_text="<b>Revenue (in Rs Thousand)</b>", secondary_y=False,range=[0,max(data_y1_new)+20])
        fig.update_yaxes(title_text="<b>Profitability (%)</b>",secondary_y=True,tickformat= ',.0%')
 
        st.plotly_chart(fig,width=1000,height=400)
        
        with st.expander("See Details"):
            col1,col2 = st.columns([2,1])
            with col1:
                st.write(HTML(financials_df[financials_df['Idea']==offer_financial].drop('Idea',axis=1).to_html(index=False,justify='center')))
            with col2:
                
                opex = ""
                i=0
                for cost_head in list(financials_df_cost_heads['Opex'][financials_df_cost_heads['Idea']==offer_financial].dropna()):
                    if i==0:
                        opex=cost_head
                        i=1
                    else:
                        opex = opex + ", "+cost_head
                opex = "Opex Cost Heads- "+opex
                        
                capex = ""
                i=0
                for cost_head in list(financials_df_cost_heads['Capex'][financials_df_cost_heads['Idea']==offer_financial].dropna()):
                    if i==0:
                        capex=cost_head
                        i=1
                    else:
                        capex = capex + ", "+cost_head
                capex = "Capex Cost Heads- "+capex
                
                st.markdown(f"<h4 style='text-align: left; color:black'>{opex}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: left; color:black'>{capex}</h4>", unsafe_allow_html=True)
            
                         
    
    #MACHINE LEARNING MODELS    
    st.markdown('---')
    #st.header('Machine Learning Model Selection')
    st.markdown(f"<h1 style='text-align: left; color: black;'>Machine Learning Model Selection</h1>", unsafe_allow_html=True)
    at3 = st.selectbox('Type of Offering', offerings)
    st.write("Selected Machine Learning Model based on Accuracy - "+log[at3].iloc[0][0])
    st.write(log[at3])
    
             

def Predict_Offerings(model,pred):
    result = predict_model(estimator = model,data = pred, raw_score =True)
    print(result)
    
def Offerings_Input(input_df,training_df,input_df_cleaned_for_prediction,offerings):
    name_list = list(input_df['10 Outlet Name'])
    city_list = list(input_df['6 City of Interview'])
    
    combined_list=[]
    for i in range(0,len(name_list)):
        combined_list.append(name_list[i]+"-"+city_list[i])
    combined_list.append("Custom Simulation")
    
    questions = {
        'Select Store': combined_list,
        'Type-of-Outlet': ['Urban', 'Rural', 'Highway'], 
    }
    
    pred = {}
    retail_intensity = ['44 Approx. area (sq. feet)','Grocery / Hypermart - Concentration of Grocery / Hypermart With respect to Population',
       'Hospital - Concentration of Hospital With respect to Population',
       'Restaurant - Concentration of Restaurant With respect to Population',
       
       'Higher Education Institutes - Concentration of higher education institutes  With respect to Population',
       
       'Apparel Brands Budget - Concentration of apparel brands budget  With respect to Population',
       'Apparel Brands Mid Range - Concentration of apparel brands mid range  With respect to Population',
       'Apparel Brands - Luxury - High Range - Concentration of apparel brands - luxury - high range With respect to Population',
       'Electronics Store - Concentration of Electronics Store  With respect to Population',
       'Gym - Concentration of Gym  With respect to Population',
       'Gym Mid-Range - Concentration of gym mid-range  With respect to Population',
       'Police Station - Concentration of Police Station  With respect to Population',
       'Pet Store - Concentration of Pet Store  With respect to Population',
       'Car Rental - Concentration of Car Rental  With respect to Population',
       'Clinics - Concentration of Clinics  With respect to Population',
       'Dentist - Concentration of Dentist  With respect to Population',
       'Pharmacy - Concentration of Pharmacy  With respect to Population',
       'Private Sector Bank - Concentration of private sector bank  With respect to Population',
       'Public Sector Bank - Concentration of public sector bank  With respect to Population',
       'Cafe - Concentration of Cafe  With respect to Population']
    
    demographic_profile = ['Demographics - Population - Total', 'Demographics - Male %',
       'Demographics - Female %', 'Demographics - Literacy %']
    
     
    st.sidebar.markdown('# Select Store')
    at1 = st.sidebar.selectbox('', questions['Select Store'])
    
    list_index= combined_list.index(at1);
    if at1 == "Custom Simulation":
        coordinates = -1
    else:
        coordinates = list_index
    
    
    st.sidebar.markdown('# Choose Your Parameters')
    st.sidebar.markdown(f"<h2 style='text-align: left; color: black;'>Type of Outlet</h2>", unsafe_allow_html=True)
    if at1 == "Custom Simulation":
        at2 = st.sidebar.selectbox('', questions['Type-of-Outlet'])
        
    else:
        at2= input_df[input_df_cleaned_for_prediction.columns[0]].iloc[list_index]
        st.sidebar.markdown(f"<h2 style='text-align: center; color: #fb843b;'>{at2}</h2>", unsafe_allow_html=True)
    
    
    add_element(pred,input_df_cleaned_for_prediction.columns[0],at2);
 
        
    st.sidebar.markdown('---')
    st.sidebar.markdown(f"<h1 style='text-align: left; color: black;'>Idea Potential</h1>", unsafe_allow_html=True)
    
    st.sidebar.markdown(f"<h2 style='text-align: left; color: black;'>Retail Intensity</h2>", unsafe_allow_html=True)
  
    val = 0
   
    if at1 == "Custom Simulation":
        
        for col in training_df[retail_intensity].columns:
            
            if training_df[col].max()<20:
                val = st.sidebar.slider(
                col,
                #min_value=training_df[col].min(),
                #max_value=training_df[col].max(),
                min_value=float(0.0),
                max_value=float(training_df[training_df['7 Type of Outlet']==at2][col].max()),
                value=float(training_df[training_df['7 Type of Outlet']==at2][col].mean()),
                step= 0.0001
                
            )
            
            else:
                val = st.sidebar.slider(
                col,
                min_value=int(round(training_df[training_df['7 Type of Outlet']==at2][col].min(),0)),
                max_value=int(round(training_df[training_df['7 Type of Outlet']==at2][col].max(),0))+1,
                
                value=int(round(training_df[col].mean(),0)),
                step = 1
            )
            
            add_element(pred,col,val) 
         
    else:
        
        for col in input_df[retail_intensity].columns:
            st.sidebar.write(col)         
            round_val = round(input_df[col].iloc[list_index],2)
            st.sidebar.markdown(f"<h2 style='text-align: center; color: #fb843b;'>{round_val}</h2>", unsafe_allow_html=True)      
            val = input_df[col].iloc[list_index]

            add_element(pred,col,val) 
   
    st.sidebar.markdown('---')
    st.sidebar.markdown(f"<h2 style='text-align: left; color: black;'>Demographic Profile</h2>", unsafe_allow_html=True)
    val=0
    if at1 == "Custom Simulation":
        
        for col in training_df[demographic_profile].columns:
            
            if training_df[col].max()<20:
                val = st.sidebar.slider(
                col,
                #min_value=training_df[col].min(),
                #max_value=training_df[col].max(),
                min_value=0.0,
                max_value=float(training_df[col].max()),
                value=float(training_df[col].mean()),
                step = 0.0001
                
            )
            
            else:
                val = st.sidebar.slider(
                col,
                min_value=int(round(training_df[col].min(),0)),
                max_value=int(round(training_df[col].max(),0))+1,
                
                value=int(round(training_df[col].mean(),0)),
                step = 1
            )        
            add_element(pred,col,val) 
         
    else:
        
        for col in input_df[demographic_profile].columns:
            st.sidebar.write(col)         
            round_val = round(input_df[col].iloc[list_index],2)
            st.sidebar.markdown(f"<h2 style='text-align: center; color: #fb843b;'>{round_val}</h2>", unsafe_allow_html=True)      
            val = input_df[col].iloc[list_index]

            add_element(pred,col,val) 

    st.sidebar.markdown('---')
    st.sidebar.markdown(f"<h1 style='text-align: left; color: black;'>Ease of Implementation</h1>", unsafe_allow_html=True)
    
    dot1 = st.sidebar.slider(
                "Dealer's Ability to Execute",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step= 0.0001
                )
    dot2 = st.sidebar.slider(
                "Area available for Idea Execution (sq ft)",
                min_value=0,
                max_value=40000,
                value=5000,
                step= 1
                )
    
    
    print (pred)
    return pred,coordinates

def save_to_pickle (file,path):
    output = open(path, 'wb')
    pickle.dump(file, output)
    output.close()

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass

def save_to_pickle_dict (file,path):
    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    

def read_from_pickle_dict(path):
    with open(path, 'rb') as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data 

def distance_calculator(input_df, training_df,API_key,offerings):
     
    gmaps = googlemaps.Client(key=API_key) 
    
    input_distance_dict= {}
    for k in input_df.index:
        distance_df = training_df.copy()
        distance_df['Distance (in KM)'] = ""
        
        distance_list = []
        source = input_df['Address'].iloc[k]   
        
        for i in distance_df.index:
            
            destination = distance_df['Address'].iloc[i]
            
            distance_dict = gmaps.distance_matrix(source, destination, mode='walking')
            if distance_dict['rows'][0]['elements'][0]['status']=='ZERO_RESULTS':
                distance_list.append(100000)
            else:                
                distance_list.append(round(distance_dict["rows"][0]["elements"][0]["distance"]["value"]/1000,2))
                
        
        distance_df['Distance (in KM)'] = distance_list

        distance_df.sort_values(by="Distance (in KM)", ascending=True, inplace = True)
        
        df_to_print = distance_df[distance_df['8 Company']!='IOCL '][['6 City of Interview','8 Company','10 Outlet Name','Address','Distance (in KM)','Offerings']].head().copy()
        
        
        
        
        df_to_print.columns = ['City', 'Company','Outlet Name','Address','Distance in KM','Offerings']
        df_to_print.reset_index(inplace = True)
        print(df_to_print)
        input_distance_dict[k]=df_to_print[['City', 'Company','Outlet Name','Address','Distance in KM','Offerings']]
    
    return input_distance_dict

def distance_calculator_custom(source, training_df,API_key,offerings):
     
    gmaps = googlemaps.Client(key=API_key)
       
    
    distance_df = training_df.copy()
    distance_df['Distance (in KM)'] = ""
    
    input_distance_dict= {}

    
    distance_list = []
    for i in distance_df.index:
        
        destination = distance_df['Address'].iloc[i]
          
        
        distance_dict = gmaps.distance_matrix(source, destination, mode='walking')
        if distance_dict['rows'][0]['elements'][0]['status']=='ZERO_RESULTS':
            distance_list.append(100000)
        else:                
            distance_list.append(round(distance_dict["rows"][0]["elements"][0]["distance"]["value"]/1000,2))
    
    
    
    distance_df['Distance (in KM)'] = distance_list

    distance_df.sort_values(by="Distance (in KM)", ascending=True, inplace = True)
    
    df_to_print = distance_df[distance_df['8 Company']!='IOCL '][['6 City of Interview','8 Company','10 Outlet Name','Address','Distance (in KM)','Offerings']].head().copy()
    
    
       
    
    df_to_print.columns = ['City', 'Company','Outlet Name','Address','Distance in KM','Offerings']
    df_to_print.reset_index(inplace = True)
    print(df_to_print)
    input_distance_dict[0]=df_to_print[['City', 'Company','Outlet Name','Address','Distance in KM','Offerings']]
    
    return input_distance_dict
        

    
if __name__ == '__main__':
    
    logging.basicConfig(level=logging.CRITICAL)
    input_df = pd.read_csv("./Files/Test Data Pivoted.csv")
    training_df = pd.read_csv("./Files//Offering Data Pivoted.csv")
    financials_df = pd.read_csv("./Files/Financials_data.csv")
    financials_df_cost_heads = pd.read_csv("./Files/Financials_Data_Cost_Heads.csv")
     
    pkl_file = open('./Files/Trained_Models_v1', 'rb')
    trained_models_pickle = pickle.load(pkl_file)
    pkl_file.close()
    
    
    log = pd.read_pickle('./Files/pickle_file.pkl')
     
  
     
    fields_to_remove = ['6 City of Interview','8 Company','10 Outlet Name','43 - NFR Present or Not', '44 Area bracket (sq. feet)','Monthly Revenue (FR)','Rev per Sq. Ft (FR)','Margin per Sq. Ft (FR)','Monthly Revenue (NFR)','NFR revenues as % of Total revenues','Rev per Sq. ft (NFR)','Margin per Sq. Ft (NFR)', 'Monthly NFR Profit','Cost as % of Revenue','Demographics - Male Population','Demographics - Female Population','Demographics - Total Literate Population','Latitude','Longitude', 'Address']
    offerings = ['Aggregated vehicle related requirements', 'Virtual C-store',
        'Truck Stops (Auto Repair / Rest)',
       'QSR / Restaurant / Dhaba',
       'One stop shop for daily farmer needs','Forecourt Advertising','Pharmacy']

    
    input_df_cleaned_for_prediction = input_df.copy()
    input_df_cleaned_for_prediction.drop(fields_to_remove,axis=1,inplace=True)
    
    trained_models = {}
    input_df_cleaned_for_prediction.drop(offerings,axis=1,inplace=True)
    for offering in offerings:
        trained_models[offering]=load_model('./Files/'+offering.split(" ")[0])
 
    # df = load_data()
    API_key = 'AIzaSyDA02-tlDUa0fH2-0V9Yo0gyyiVTLA21Zc'#enter Google Maps API key
    
    training_df['Offerings']=""
    for i in training_df.index:
        offering_list = []
        
        for offering in offerings:
            
            
            if training_df[offering].iloc[i]==1:
                offering_list.append(offering)
        
        training_df['Offerings'].iloc[i] = offering_list
                
    input_distance_dict = {}
    #input_distance_dict = distance_calculator(input_df, training_df, API_key,offerings)
        
        
    #save_to_pickle_dict(input_distance_dict, './Files/input_distance.pkl')
    input_distance_dict = read_from_pickle_dict('./Files/input_distance.pkl')
                   

    
    
    title_placeholder = st.sidebar.empty()
    
              
    main(input_df,training_df,input_df_cleaned_for_prediction,offerings,trained_models,log,input_distance_dict,API_key,financials_df,financials_df_cost_heads)
