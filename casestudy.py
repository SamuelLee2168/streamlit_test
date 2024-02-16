import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plots
plots.style.use("fivethirtyeight")
import plotly.graph_objects as plot
import plotly.express as px
from PIL import Image
import time


def standard_units(any_numbers):
    return (any_numbers - np.average(any_numbers)) / np.std(any_numbers)

# Below, t is a table; x and y are column indices or labels.

def correlation(t, x, y):
    return np.mean(standard_units(np.array(t.loc[:,x])) * standard_units(np.array(t.loc[:,y])))

def slope(t, x, y):
    r = correlation(t, x, y)
    return r * np.std(np.array(t.loc[:,y])) / np.std(np.array(t.loc[:,x]))

def intercept(t, x, y):
    return np.mean(np.array(t.loc[:,y])) - slope(t, x, y) * np.mean(np.array(t.loc[:,x]))

def fitted_values(t, x, y):
    a = slope(t, x, y)
    b = intercept(t, x, y)
    return a * t.loc[:,x] + b
def plot_fitted(t, x, y):
    tbl = t.select(x, y)
    tbl.with_columns('Fitted Value', fitted_values(t, x, y)).scatter(0)

    
testimg = Image.open("apple.jpg")

st.title("World happyness case study!")

#conda activate plotlyx
#streamlit run casestudy.py

happy = pd.read_csv("world-happiness-report.csv")
happy
st.info("System loading casestudy.py")
#st.info("Information")
#st.warning("Warning")

# Text
st.write("Above is the perpendicular bisector of the photosynthesis pentameter's quantum fluctuations to the binomial distribution of the pathetic fallacy from the constant acceleration of the WORLD HAPPYNESS TABLE!!!")
# Header
st.header("1. Life ladder's correlation between other stuff")

# Image
st.image(testimg, width=128)
  
#Check
if st.checkbox("Show results?❄︎☟"):
    Ladder_Corr = happy.corr()
    Ladder_Corr
    st.write("Next, we are going to see the top 3 values that correlate the most with the life ladder!")
    Ladder_Corr.sort_values("Life Ladder", ascending = False)
    Ladder_Corr
    st.write("So basically as you can see the answer is Life Ladder, Log GDP per capita, Healthy life expentancy at birth, and Social support")
    def scatter(x,y):
      fig = px.scatter(happy, x=x, y=y)
      st.plotly_chart(fig)
      time.sleep(1)
    
    st.write("Now we are going to create some scatter plots with this!")
    scatter("Life Ladder","Log GDP per capita")
    st.write("Nice positive correlation right? We have more!")
    scatter("Life Ladder","Healthy life expectancy at birth")
    scatter("Life Ladder","Social support")
    st.write("Now we hvae explored the positive correlations, we should take a look at the negative! Here:")
    Ladder_Corr.sort_values("Life Ladder", ascending = True)
    Ladder_Corr
    st.write("As you can see, Perceptions of corruption and Negative affect has the move negative correlation with life ladder!!")
    st.write("Time to plot some scatters")
    scatter("Life Ladder","Perceptions of corruption")
    scatter("Life Ladder","Negative affect")
    st.write("Ok. After this, lets get the top 5 of the happiest countries! Like this!")
    st.markdown("grouped = happy.groupby(['Country name']).mean().sort_values(""!Life Ladde"r", ascending = False)print(grouped.head(5))")
    grouped = happy.groupby(['Country name']).mean().sort_values("Life Ladder", ascending = False)
    st.text(grouped.head(5))
    st.write("here comes the countries that has the lowest life ladder :(")
    st.text(grouped.tail(5))
    
# slider

# first argument takes the title of the slider
# second argument takes thr starting of the slider
# last argument takes the end number
st.write("Hello! This is a prediction model!")
rating = st.slider("Please input how generous you think your country is!!! :) ", min_value=-1.0, max_value=1.0, step=0.001)

st.write("This is a test: ")
st.markdown(happy.loc[:,"Life Ladder"])
st.markdown(happy.loc[:,"Generosity"])
st.markdown(np.array(happy.loc[:,"Life Ladder"]))
st.markdown(np.array(happy.loc[:,"Generosity"]))

# print the level
# format() is used to print value
# of a variable at a specific position
st.text('Selected: {}'.format(rating))
slope = slope(happy, "Generosity", "Life Ladder")
intercept = intercept(happy, "Generosity", "Life Ladder")
st.write("Life Expectancy: " + slope * rating + intercept)
print("Life Expectancy: " + slope * 0.5 + intercept)
# Subheader
st.subheader("Why are you looking at this??? read other stuff lol this is useless...")

