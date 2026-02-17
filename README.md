# olympics classification project

How to run (terminal commands in project directory): 

First time:
    python -m venv venv
    venv/Scripts/Activate
    pip install -r requirements.txt

Before push:
    pip freeze > requirements.txt

Subsequent times (after pulls):
    venv/Scripts/Activate
    pip install -r requirements.txt