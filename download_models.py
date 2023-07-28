import os
import urllib.request
import streamlit as st

def download_model_if_not_exist(url, file_name, models_dir="./models"):
    """
    Download a model file from the given URL to the specified directory if it does not exist.

    Parameters:
        url (str): The URL of the model file.
        file_name (str): The name of the model file.
        models_dir (str): The directory to save the downloaded model file.

    Returns:
        bool: True if the model was downloaded, False if it already exists.
    """
    if not os.path.exists(os.path.join(models_dir, file_name)):
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        st.sidebar.success("Downloading the model...")
        urllib.request.urlretrieve(url, os.path.join(models_dir, file_name))
        st.sidebar.success("Download complete.")
        st.experimental_rerun()
        return True
    return False
