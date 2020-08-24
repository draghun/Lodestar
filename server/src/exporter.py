
from IPython.nbformat import v3, v4

from codeblock_class import CodeBlock
import inspect
import ujson 
import os
import json

class Exporter():
    def __init__(self):
        self.path_file = ""
        self.codeblocks = CodeBlock()
        self.notebook_tuple = []

        load_method = 'load_dataset'
        load_des = 'Load your dataset here'
        method_to_call = getattr(CodeBlock, load_method)
        load_dataset = inspect.getsource(method_to_call)
        self.notebook_tuple.append([load_method, load_des, load_dataset])

        save_method = 'save_bytes_image'
        save_des = 'Save bytes image to image list'
        method_to_call = getattr(CodeBlock, save_method)
        save_image = inspect.getsource(method_to_call)
        self.notebook_tuple.append([save_method, save_des, save_image])

        performance_method = 'performance_metric'
        performance_des = 'Calculates and returns the performance score between true and predicted values based on the metric chosen'
        method_to_call = getattr(CodeBlock, performance_method)
        performance_metric = inspect.getsource(method_to_call)
        self.notebook_tuple.append([performance_method, performance_des, performance_metric])

        fit_method = 'fit_model'
        fit_des = 'Using GridSearchCV to find optimal model after the fitting the data'
        method_to_call = getattr(CodeBlock, fit_method)
        fit_model = inspect.getsource(method_to_call)
        self.notebook_tuple.append([fit_method, fit_des, fit_model])

        kmeans_method = 'initializeClustersForKmeans'
        kmeans_des = 'Initializing clustering for the kmean cluster algorithm'
        method_to_call = getattr(CodeBlock, kmeans_method)
        kmeans_clust = inspect.getsource(method_to_call)
        self.notebook_tuple.append([kmeans_method, kmeans_des, kmeans_clust])

        calc_method = 'calcConditionalFreqDist'
        calc_des = 'Caclulating conditional frequency distribution'
        method_to_call = getattr(CodeBlock, calc_method)
        calc_freqdist = inspect.getsource(method_to_call)
        self.notebook_tuple.append([calc_method, calc_des, calc_freqdist])

        wordvec_method = 'calcWordVec'
        wordvec_des = 'Calculating work vector'
        method_to_call = getattr(CodeBlock, wordvec_method)
        calc_wordvec = inspect.getsource(method_to_call)
        self.notebook_tuple.append([wordvec_method, wordvec_des, calc_wordvec])

    def download_notebook(self, current_analysis):
        string = "# -*- coding: utf-8 -*-\n# <nbformat>3.0</nbformat>\n"

        other_tuples = self.get_notebook_tuples(current_analysis)
        for other in other_tuples:
            self.notebook_tuple.append(other)

        for tup in self.notebook_tuple:
            print(tup)
            string += "# <markdowncell>\n"
            string += "Method: " + str(tup[0]) + '\n'
            string += "# <markdowncell>\n"
            string += "Descripton: " + str(tup[1]) + '\n'
            string += "# <codecell>\n"
            string += str(tup[2]) + '\n'

        nbook = v3.reads_py(string)
        nbook = v4.upgrade(nbook)
        jsonform = v4.writes(nbook) + '\n'

        print(jsonform)

        return jsonform

    def get_notebook_tuples(self, notebook_methods):
        notebook_tuples = []
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary_code_map.json', 'r') as f_dict:
            json_dict = json.load(f_dict)

        for method in notebook_methods:
            for i in range(0, len(json_dict)):
                if method == json_dict[i]['user-data']['method']:
                    description = json_dict[i]['description']
                    method_to_call = getattr(CodeBlock, json_dict[i]['method'])
                    code_string = inspect.getsource(method_to_call)
                    notebook_tuples.append([method, description, code_string])
                    break

        return notebook_tuples

if __name__ == "__main__":
    exporter = Exporter()
    exporter.download_notebook(['first10samples'])
