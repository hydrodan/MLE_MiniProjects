Model deployment instructions
______________________
1. Run `capstone_step5_EDA_Pipeline` to generate the Pipeline and Preprocessing objects & files, as well as the `X.pkl`
data file containing imputed metadata. (~2 hours)
2. Run `capstone_step7_model_experimentation` to generate the Tuned model (~11 hours)
3. Copy the created `capstone_X.pkl` and `model.joblib` files to the deployment/model folder for deployment.
4. Running the `deployment.py` script will start a Flask instance, which can be browsed to in a webbrowser to interact
with the model.

Note: model files are not included in git because they will need to be version specific to whatever your python version
is locally.