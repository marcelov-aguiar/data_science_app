import os,sys
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage
from pages import data_upload, data_exploration, change_metadata, machine_learning, \
    dashboard

from utils import utils

# Create an instance of the app 
app = MultiPage()

# Title of the main page
display = Image.open(os.path.abspath(os.path.join(sys.path[0],'..' ,'img','uerj_logo.jpg')))
display = np.array(display)

col1, col2 = st.columns(2)
col1.image(display, width = 400)
col2.title("Data Science Application")

# Add all your application here
app.add_page("Upload Data", data_upload.app)
app.add_page("Change Metadata", change_metadata.app)
app.add_page("Data Exploration",data_exploration.app)
app.add_page("Machine Learning", machine_learning.app)
app.add_page("Dashboard",dashboard.app)

# The main app
app.run()
