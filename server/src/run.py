'''Backend application runner	
'''

from flask import Flask, request, Response, send_file, jsonify

import os
import pandas as pd
import numpy as np
import ujson
import json
import time
import logging

from analysis import Analysis
from recommender import Recommender
from exporter import Exporter

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.disabled = True

exporter = Exporter()
analysis = Analysis()
recommender = Recommender()
data_read = pd.DataFrame()
chosen_analysis = []

'''	
Input Request:      None	
Description:        Returns list of stored datasets.	
Output Response:    List containing stored datasets in JSON	
'''
@app.route('/datasets')
def scan_dataset():
    global chosen_analysis

    data_path = os.path.dirname(os.path.abspath(__file__)) + '/../../data/'
    files = [f.split('.')[0] for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    print('datasets requested')
    return ujson.dumps(files)

'''	
Input Request:      dataset: string: Selected dataset
dataset is what the uder chooses to input
python in backend -> pick up dataset and load it into the dataframe
data anyalsis -> what the user chooses to do
output data frame -> what comes out of it

Description:        Load selected dataset into dataframe to be used for analysis	
Output Response:    None	
'''
@app.route('/dataload')
def data_load():
    global analysis
    global recommender
    global data_read
    global chosen_analysis

    analysis = Analysis()
    recommender = Recommender()
    data_read = pd.DataFrame()
    chosen_analysis = []

    if (request.args["type"] == "upload"):
        with open('../../data/uploadedData.csv', 'wb') as fp:
            fp.write(request.args['dataset'])
            print("Wrote into uploadedData.csv")

        data_read = pd.read_csv('../../data/uploadedData.csv')
    else:
        data_read = load_dataset(request.args['dataset'])

    return 'Data Loaded!'

'''	
Input Request:      None	
Description:        The user selects what they want to compute. 
                    If the chosen computation is a valid one, then it computes it and then outputs ____ 	
Output Response:    ujson -> converts string result into a json object
	dictionary is converted to json object
    getting passed back and forht is strings
'''
@app.route('/compute')
def compute():
    global chosen_analysis

    print('chosen_analysis: %s' % str(chosen_analysis))

    if request.args['compute'] == 'suggest':
        state = None if not request.args['state'] else request.args['state']
        dataset = None if not request.args['dataset'] else request.args['dataset']
        print ("state: ")
        print (state)
        print ("type of dataset: ")
        print (dataset)
        expert_suggestions = recommender.get_manual_suggestions(dataset, state)
        crowd_suggestions = recommender.get_crowd_suggestions(dataset, state)

        all_analysis = recommender.get_analysis_list()
        res = {
            'all_analysis': all_analysis,
            'manual_result' : expert_suggestions,
            'crowd_result' : crowd_suggestions,
            'chosen_analysis' : chosen_analysis,
            'type' : 'suggest'
        }
        return ujson.dumps(res)

    else:
        analysis.intermediate_selected = True

        if len(analysis.intermediate_df) != 0:
            analysis.current_df = analysis.intermediate_df[-1]

        chosen_analysis.append(str(request.args['compute']))

        result = analysis.execute_analysis(request.args['compute'], request.args['dataset'], request.args['state'])
        result['chosen_analysis'] = str(chosen_analysis)

        return ujson.dumps(result)

'''	Input Request:      index: integer	
                        Index of the dataframe to be deleted/exported	
                    type: string	
                        Desired action on dataframe, either 'delete' or 'export'        	
Description:        	
					If the requested action is 'delete': Deletes all intermediate dataframes at and after the selected dataframe	
                    If the requested action is 'export': Exports selected intermediate dataframe	
Output Response:    Type of action taken, either 'delete' or 'export', in JSON  	
'''
@app.route('/intermediatedata')
def delete():
    global chosen_analysis

    res = {
        'type': request.args['type']
    }
    if request.args['type'] == 'delete':
        index = request.args['index']
        analysis.delete_intermediate_df(index)
        ind = int(index) - 1
        chosen_analysis = chosen_analysis[:ind]

    elif request.args['type'] == 'export':
        ind = int(index) - 1
        res['data'] = analysis.export_intermediate_df(ind).to_json(orient='records')

    return ujson.dumps(res)

'''	
Input Request:      None	
Description:        Exports the notebook after it computes the user's selection. 
                    Notebook gets downloaded into the user's computer
Output Response:    unsure	
'''
@app.route('/export')
def export():
    global chosen_analysis
    global exporter

    try:
        jsonform = exporter.download_notebook(chosen_analysis)

        res = {
            'status': 'success',
            'data': jsonform
        }
    except Exception as e:
        res = {
            'status': 'error',
            'data' : str(e)
        }

    return ujson.dumps(res)

'''	
Input Request:      None	
Description:        Flask function
Output Response:    unsure	
'''
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

'''	
Input Parameters:   filename: string	
                        Filename of dataset to be loaded	
Description:        Helper for data_load(). Load dataset into dataframe.	
Output:             df: Pandas dataframe	
                    	Dataframe with dataset loaded onto it	
'''
def load_dataset(filename):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) + '/../../data/', filename + ".csv")
    df = pd.read_csv(data_path)
    return df


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
