''' The Analysis class primarily handles executing the selected analysis methods using codeblocks, and also handles 
    deleting and exporting dataframes.
'''
import matplotlib
matplotlib.use('agg')

from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
import seaborn
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import io
import json
import base64
import itertools
from codeblock_class import CodeBlock
import inspect

class Analysis():

    '''
    Input Parameters:   None
    Description:        Initialize fields
    Output:             None
    '''
    def __init__(self):
        self.intermediate_df = []
        self.loaded_dataset = pd.DataFrame()
        self.current_df = pd.DataFrame()
        self.dataset = ""
        self.intermediate_selected = False

    '''
    Input Parameters:   dataset: string
                            Filename of dataset to be loaded  
    Description:        Load passed in dataset into the dataframe
    Output:             None
    '''
    def dataset_changed(self, dataset):
        self.dataset = dataset
        self.loaded_dataset = load_dataset(dataset)
        self.intermediate_df = []
        self.current_df = []

    '''
    Input Parameters:   index: string of an integer
                            index of the intermediate dataframe
    Description:        Delete all dataframes at and after the given index
    Output:             None
    '''
    def delete_intermediate_df(self, index):
        ind = int(index) - 1
        self.intermediate_df = self.intermediate_df[:ind]
        print('length is ' + str(len(self.intermediate_df)))

    '''
    Input parameters:   index: string of an integer
                            index of the intermediate dataframe
    Description:        Exports intermediate dataframe at the given index   
    Output:             
                        Dataframe at requested index, or new Dataframe if index out of range
    ''' 
    def export_intermediate_df(self, index):
        idx = int(index)
        if idx < len(self.intermediate_df):
            return self.intermediate_df[idx]
        else:
            return pd.DataFrame()

    '''
    Input parameters:   method: string
                            Name of the codeblock to execute
                        dataset: string
                            Filename of the dataset
                        state: JSON string
                            Current application state based on current state of program
    Description:        Executes the given codeblock on the dataframe
    Output:             res: dictionary
                            Dictionary containing head of loaded dataset, dataframe resulting from executing the codeblock, description of the executed codeblock, and flag stating that it is an analysis method 
    '''
    def execute_analysis(self, method, dataset, state=None):
        if self.dataset != dataset:
            self.dataset_changed(dataset)

        if len(self.intermediate_df) == 0:
            self.loaded_dataset = load_dataset(dataset)
            self.current_df = self.loaded_dataset

        description = ''
        json_data = {}
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary_code_map.json', 'r') as f:
            json_data = json.load(f)

        method_label = ''
        for data in json_data:
            if method == data["label"]:
                method_label = data['method']
                description = data['description']
                break

        c = CodeBlock()
        method_to_call = getattr(CodeBlock, method_label)
        res = method_to_call(c, self.loaded_dataset, self.intermediate_df, description, method)

        res['code'] = inspect.getsource(method_to_call)

        return res

'''
Input Parameters:   filename: string
                        Filename of dataset to be loaded
Description:        Helper function to load dataset into dataframe.
Output:             df: Pandas dataframe
                        Dataframe with dataset loaded onto it
'''
def load_dataset(filename):
    data_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)) + '/../../data/', filename + ".csv")
    df = pd.read_csv(data_path)
    return df