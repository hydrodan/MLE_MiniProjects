# link to tutorial that explains what's going on : https://www.geeksforgeeks.org/retrieving-html-from-data-using-flask/#
# test wells: 0402912228

from flask import Flask, render_template, request
import threading
import matplotlib.pyplot as plt
import io
import os
import base64
from perlin_noise import PerlinNoise

# for the model deployment
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

# set up the fail case where the script can exit out
class StopExecution(Exception):
    def _render_traceback_(self):
        pass

def exit_notebook(): raise StopExecution

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV
from sklearn.linear_model import LogisticRegression

# some debugging stuff
print(os.getcwd())
assert( os.path.exists('./template'))
assert( os.path.exists('./template/index.html'))

os.environ["FLASK_DEBUG"] = "development"

app = Flask(__name__, template_folder='./template', static_folder='./static')
app.Debug = True
app.config["EXPLAIN_TEMPLATE_LOADING"] = True # get more debug output

# open port 35000 in the firewall & port-forward if needed
port = 35000
public_url = "0.0.0.0"
app.config["BASE_URL"] = public_url

### Model loading
model = joblib.load('model/tuned_model.joblib')
print(model)

# load the pickle file for the source dataset
pickle_x_file_name = "model/capstone_x.pkl"
if os.path.exists(pickle_x_file_name) :
    print("Reading existing well start and end dates pickle...")
    X_df = pd.read_pickle(pickle_x_file_name)
else:
    print("Can't locate pickle files in current directory; try running/updating the capstone step 5 EDA first to generate the pickle files")
    exit_notebook()

# Check that the data loaded ok
print(X_df.head())
print(X_df.columns)
print(X_df.loc['0402912228'])

# This runs the model after the user POSTs an API number
def run_model(api_list):
    # found a Py3 tutorial here: https://stackoverflow.com/questions/38061267/matplotlib-graphic-image-to-base64/63381737

    results_list = []
    for api in api_list:
        if api[0:2] == "04" and len(api) == 10:
            pred_x = X_df.loc[[api]] # extracts the original features for this well
            #pred_x = model['preprocessor'].transform(pred_x)  # runs the preprocessing step
            y_pred = model.predict(pred_x)
            y_proba = model.predict_proba(pred_x)
            y_conf = max(y_proba[0])

            results = {
                'api10' : api,
                'prediction': y_pred,
                'probabilities' : y_proba[0],
                'prediction_string' : "Not at risk" if y_pred == 0 else "At risk of fines or orphaning",
                'confidence': y_conf,
                'county': pred_x['CountyName'].values[0],
                'operator_size': pred_x['OperatorBin'].values[0],
                'idle_year': pred_x['CalcIdleYear'].values[0],
                'spud_year': pred_x['SpudYear_Model'].values[0],
                'well_status': pred_x['WellStatus'].values[0],
                'well_type': pred_x['WellType'].values[0],
            }
            results_list.append(results)
        else:
            print("Malformed API number : expected a 10 digit string, prefixed by '04' followed by a legitimate well identifier.")

    return results_list

# generates a graphic
def generate_bar_chart(values, categories, title):
    # Write code here for a function that takes a list of category names and
    # respective values
    # found a Py3 tutorial here: https://stackoverflow.com/questions/38061267/matplotlib-graphic-image-to-base64/63381737

    print(categories)
    print(values)

    fig, ax = plt.subplots()
    ax.bar(categories, values, edgecolor='white')
    ax.set_ylim(0,1)
    ax.set_ylabel('Confidence')
    plt.suptitle(title)
    plt.tight_layout()
    iob = io.BytesIO()
    plt.savefig(iob, format='png')
    plt.close()
    iob.seek(0)
    mime_type = "image/png"  # same as the plt.savefig() function above
    base64_enc = base64.b64encode(iob.read()).decode('utf-8') # decode converts binary data back to text for the browser
    base64_pic = f"data:{mime_type};base64,{base64_enc}"
    return base64_pic

@app.route('/', methods=['GET', 'POST'])
def index():
    result_list = None
    bar_chart = None

    # categories for the bar chart
    prob_categories = ['Okay', 'At Risk']

    if request.method == 'POST':
        # Extract categories from the request form and convert the string to a
        # list.
        apis = request.form.get("api10")
        if apis != None:
            api_list = list(map(str.strip, apis.split(",")))  # split by comma and strip whitespace
            # Pass your categories and values to the generate_bar_chart function.
            result_list = run_model(api_list)

            # if there is a result, build a bar chart of the probability results
            if result_list != None:
                for result in result_list:
                    bar_chart = generate_bar_chart(result['probabilities'], prob_categories, result['api10'])

    # Return a render_template function, passing your bar plot as input.
    return render_template("index.html", results=result_list, bar_chart=bar_chart)

#
# Main functionality that starts the Flask server from the command line
#

if __name__ == '__main__':
  # Start the Flask server in a new thread
  #  app.run(host='0.0.0.0', debug=True, port=35000)
  threading.Thread(target=app.run, kwargs={"use_reloader": False, 'port':port, 'host':'0.0.0.0'}).start()