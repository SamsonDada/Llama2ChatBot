## Installation

1. Setting Up the Virtual Environment:
   - Open a terminal or command prompt.
   - Create a virtual environment using the Python's built-in venv module with the following command:
        
        ```
        python3 -m venv chatbot
        ```
        
    This will create a new directory named 'cellvision'. Feel free to choose a different name if you prefer.

    - Activate the virtual environment with the following command:
       -  On macOS and Linux:

            ```
            source chatbot/bin/activate
            ```

       -  On Windows:

            ```
            .\chatbot\Scripts\activate
            ```

    You will know the virtual environment is activated when its name appears in your command prompt.   


2. install the necessary packages with the following command:
    ```
    pip install -r requirements.txt
    ```

3. run app
```
streamlit run app.py
```