# Llama 2 ChatBot
<p align="left">
    <img width=180 src="logo/llama-144.png">
</p>
This project is a chatbot built with Llama 2. 

## Installation

Follow these steps to install and run the chatbot:

### 1. Setting Up the Virtual Environment

1. Open a terminal or command prompt.

2. Create a virtual environment using Python's built-in `venv` module with the following command:

    ```bash
    python3 -m venv chatbot
    ```
   
    This command creates a new directory named 'chatbot'. Feel free to choose a different name if you prefer.

3. Activate the virtual environment with the following command:

    - On macOS and Linux:

        ```bash
        source chatbot/bin/activate
        ```

    - On Windows:

        ```bash
        .\chatbot\Scripts\activate
        ```

    You will know the virtual environment is activated when its name appears in your command prompt.   

### 2. Install the Necessary Packages

Once the virtual environment is activated, install the necessary packages with the following command:

```bash
pip install -r requirements.txt
```

### 3. Run the App
After all the packages have been installed, you can run the app with the following command:

```bash
streamlit run app.py
```

Enjoy interacting with the Llama 2 ChatBot!

