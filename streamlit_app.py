import streamlit as st

import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go

import sklearn 
from sklearn.ensemble import GradientBoostingClassifier
from plotly.graph_objects import Layout
import pickle

st.set_page_config(
    page_title="Rugby Kicking Prediction",
    #page_icon="👋",
)

st.write("### Here is the data I scrapped, labeled as makes and misses on the first tab, and the predictions for the testing data set on the second tab")


#Pitch diagram 
def read_in_data():
    data = pd.read_csv('data/total_data.csv')
    return data 

data = read_in_data()


def fig1(data):
    fig = px.scatter(x=data['x_meters'], y=data['y_meters'], color = data['result'], opacity= data['opacity'], 
                      title="Pitch Diagram With All Scrapped Data",
                    labels = {"color":'Make or Miss', 
                             "x":"Meters From Try Line", 
                             "y":"Meters from touch (left)"},
                       height = 600
                    )
    
    
    fig.add_shape(type="rect",
    x0=0, y0=32.5, x1=-1, y1=37.5,
    line=dict(color="black"),
    )
    fig.add_shape(type="rect",
        x0=0, y0=32.5, x1=-10, y1=37.5,
        line=dict(color="black"),
    )
    fig.add_hline(y=70, line_color="green")
    fig.add_hline(y=0, line_color="green")
    fig.add_hline(y=65,  line_dash="dashdot", line_color="green")
    fig.add_hline(y=65,   line_dash="dot", line_color="white")
    fig.add_hline(y=55,  line_dash="dashdot", line_color="green")
    fig.add_hline(y=55,  line_dash="dot", line_color="white")
    fig.add_hline(y=5, line_dash="dashdot", line_color="green")
    fig.add_hline(y=5,  line_dash="dot", line_color="white")
    fig.add_hline(y=15,  line_dash="dashdot", line_color="green")
    fig.add_hline(y=15, line_dash="dot", line_color="white")
    fig.add_vline(x=40,  line_dash="dash", line_color="green")
    fig.add_shape(type="rect",
        x0=0, y0=5, x1=-10, y1=15,
        line=dict(color="white"),
    )

    fig.add_shape(type="rect",
        x0=0, y0=55, x1=-10, y1=65,
        line=dict(color="white"),
    )


    fig.add_vline(x=0, line_color="green")
    fig.add_vline(x=-3, line_color="green")
    fig.add_vline(x=22, line_color="green")
    fig.add_vline(x=50, line_color="green")
    fig.add_vline(x=5,  line_dash="dash", line_color="green")


    fig.add_hline(y=[-3,0], line_width=3, line_color="black")
    fig.update_layout(yaxis_range=[0,70])
    fig.update_layout(xaxis_range=[-3,55])
    fig.update_layout(
        xaxis_title="Meters From Try Line",
        yaxis_title="Try Zone",
        font=dict(
            family="Times New Roman",
            size=13,
            color="#7f7f7f"
        )
    )
    fig.update_layout(
                      xaxis = dict(
                        tickmode='array', #change 1
                        tickvals = [0,5,22,40,50], #change 2
                        ticktext = ['Try Line',5,22,40,50], #change 3
                        ),
                       font=dict(size=18, color="black"), 

    )
    fig.update_layout(
                      yaxis = dict(
                        tickmode='array', #change 1
                        tickvals = [5,15,55,65], #change 2
                        ticktext = ['5', '15', '15', '5'], #change 3
                        ),

    )


    # Set templates
    fig.update_layout(template="plotly_white", )
    return fig


pred_data = pd.read_csv("data/stream_lit_preds.csv")


def heat_map(df):
    fig = px.scatter(
    x=df['x_meters'],
    y=df['y_meters'],
   color = df['probs'],
   opacity= 0.7, 
   title="Pitch Diagram With All Scrapped Data",
                    labels = {"color":'Predicted Probability of Make', 
                             "x":"Meters From Try Line", 
                             "y":"Meters from Touch (left)"},
                       height = 600,
                       width = 1000
                    )
    
    
    fig.add_shape(type="rect",
    x0=0, y0=32.5, x1=-1, y1=37.5,
    line=dict(color="black"),
    )
    fig.add_shape(type="rect",
        x0=0, y0=32.5, x1=-10, y1=37.5,
        line=dict(color="black"),
    )
    fig.add_hline(y=70, line_color="green")
    fig.add_hline(y=0, line_color="green")
    fig.add_hline(y=65,  line_dash="dashdot", line_color="green")
    fig.add_hline(y=65,   line_dash="dot", line_color="white")
    fig.add_hline(y=55,  line_dash="dashdot", line_color="green")
    fig.add_hline(y=55,  line_dash="dot", line_color="white")
    fig.add_hline(y=5, line_dash="dashdot", line_color="green")
    fig.add_hline(y=5,  line_dash="dot", line_color="white")
    fig.add_hline(y=15,  line_dash="dashdot", line_color="green")
    fig.add_hline(y=15, line_dash="dot", line_color="white")
    fig.add_vline(x=40,  line_dash="dash", line_color="green")
    fig.add_shape(type="rect",
        x0=0, y0=5, x1=-10, y1=15,
        line=dict(color="white"),
    )

    fig.add_shape(type="rect",
        x0=0, y0=55, x1=-10, y1=65,
        line=dict(color="white"),
    )


    fig.add_vline(x=0, line_color="green")
    fig.add_vline(x=-3, line_color="green")
    fig.add_vline(x=22, line_color="green")
    fig.add_vline(x=50, line_color="green")
    fig.add_vline(x=5,  line_dash="dash", line_color="green")


    fig.add_hline(y=[-3,0], line_width=3, line_color="black")
    fig.update_layout(yaxis_range=[0,70])
    fig.update_layout(xaxis_range=[-3,55])
    fig.update_layout(
        xaxis_title="Meters From Try Line",
        yaxis_title="Try Zone",
        font=dict(
            family="Times New Roman",
            size=13,
            color="#7f7f7f"
        )
    )
    fig.update_layout(
                      xaxis = dict(
                        tickmode='array', #change 1
                        tickvals = [0,5,22,40,50], #change 2
                        ticktext = ['Try Line',5,22,40,50], #change 3
                        ),
                       font=dict(size=18, color="black"), 

    )
    fig.update_layout(
                      yaxis = dict(
                        tickmode='array', #change 1
                        tickvals = [5,15,55,65], #change 2
                        ticktext = ['5', '15', '15', '5'], #change 3
                        ),

    )


    # Set templates
    fig.update_layout(template="plotly_white", )
    return fig


tab1, tab2 = st.tabs(["All Kicking Data Scrapped", "Scatter Plot of Probability on Testing Data"])
with tab1:
    # Use the Streamlit theme.
    # This is the default. So you can also omit the theme argument.
    st.plotly_chart(fig1(data), theme="streamlit", use_container_width=True)
with tab2:
    # Use the native Plotly theme.
    st.plotly_chart(heat_map(pred_data), theme=None, use_container_width=False)

    
st.write("Also go Ireland!")


y_m = float(st.number_input('How many meters in from touch are you kicking?'))


l_r = st.selectbox(

    'Left or Right Touch Line?',

    ('Left','Right',))
#displaying the selected option

x_m = float(st.number_input('How many meters in from the try line are you kicking?'))


st.write('You are in from the try', x_m,' and ',y_m,' meters in from the ',l_r,' touch line.')

if st.button('Calculate Probability '):
    st.write('Calculating')
    #start = time.time()
    #st.write("Started run time at: ", start)
    
    #import model from notebook 
    #print probability 
    #

    loaded_model = pickle.load(open('pickled_models/kicker_model.pkl', 'rb'))
    
    if l_r == 'Left':
        l_r2 = x_m
    else:
        l_r2 = 70-x_m
    
    input_df = pd.DataFrame(columns = ['x_meters', 'y_meters'],
                            data = [[x_m, l_r2]])
    
    
    array = loaded_model.predict_proba(input_df)
    prob2 = round(array[0][1], 2)
    str_2 = str(prob2).replace("0.", '')
    
    st.write('Prediction Probability: ' + str_2 +"%")
    
    

    
    
    
    
    