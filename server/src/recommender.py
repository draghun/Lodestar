'''The Recommender class determines the expert and crowd recommendations given the current place in the data analysis process.
''' 

import pandas as pd
import numpy as np
import os
import json
import random
from analysis import Analysis

crowd_analysis_order = 'new_crowd_analysis_order.json'

class Recommender:
  
    '''
    Input Parameters:   state: JSON string
                            Current state of the application
    Description:        Initialize fields
    Output:             None
    '''
    def __init__(self, state=None):
        self.state = state
        self.checked = {}

        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + crowd_analysis_order, 'r') as f:
            self.json_data = json.load(f)
            self.crowdIds = self.getCrowdIds()
            self.expertIds = self.getExpertIds()

    '''
    Input Parameters:   filtered_dict: list of dictionaries
                    List of codeblocks compatible with dataset
                state: JSON string
                current state of the application
    Description:        
                Get next set of expert suggestions given current state
    Output:             suggestions: list of dictionaries
                Each dictionary is a codeblock that is manually suggested, with the keys 'name', 'probability', and 'description'. List is ranked in descending order by probability. 
        '''
    #modify so it interacts with dictionary and returns suggestions
    #ranks list of entries its been given from input dictionary or list of templates/codeblocks
    #change this to use dictionary
    def get_manual_suggestions(self, dataset, state=None):

        print ("in get_manual_suggestions")
        print ("name of dataset: ")
        print (dataset)
        suggestion_dict = dict()
        checked = {}

        # the name of the most recently executed block
        current_analysis = json.loads(state)

        # do we have a name?
        if current_analysis:
          # get the id
          current_id = self.getIdForBlockName(current_analysis)
          # this recommendation was from the crowd, need something from the experts
          if current_id and current_id not in self.expertIds:
            # get the relevant expert ids
            candidates = self.retrieveExpertBlockIdsByBlockTags(current_id)
            # do we have any expert blocks with overlapping tags?
            if candidates and len(candidates) > 0:
              # randomly pick a block from the list for now
              candidate_count = 0
              while(True):
                rand_int = random.randint(0,len(candidates)-1)
                new_id = list(candidates)[rand_int]
                # get the corresponding block name
                current_analysis = self.codeblock_id_to_name(new_id)
                candidate_count += 1
                if (current_analysis in checked.keys() and checked[current_analysis]):
                  break
                else:
                  check = self.check_codeblock(current_analysis, dataset, state)
                  checked[current_analysis] = check
                  if (check):
                    break

                if(candidate_count == len(candidates)):
                  break

        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'manual_analysis_order.json', 'r') as f:
            json_data = json.load(f)

        for i in range(0, len(json_data)):
            tutorial_order = json_data[i]["analysis-order"]

            for index, step in enumerate(tutorial_order):
                if current_analysis:
                    if step == current_analysis and index <  len(tutorial_order) - 1:

                        if (tutorial_order[index + 1] in checked.keys()):
                          check = checked[tutorial_order[index + 1]]
                        else:
                          check = self.check_codeblock(tutorial_order[index + 1], dataset, state)
                          checked[tutorial_order[index + 1]] = check

                        if (check):
                          if (tutorial_order[index + 1] not in suggestion_dict.keys()):
                              suggestion_dict[tutorial_order[index + 1]] = [1, self.get_description(tutorial_order[index + 1])]
                          else:
                              curr_val = suggestion_dict [tutorial_order[index + 1]][0]
                              suggestion_dict[tutorial_order[index + 1]][0] = curr_val + 1
                else:
                    if (tutorial_order[0] in checked.keys()):
                      check = checked[tutorial_order[0]]
                    else:
                      check = self.check_codeblock(tutorial_order[0], dataset, state)
                      checked[tutorial_order[0]] = check

                    if (check):
                      if tutorial_order[0] not in suggestion_dict.keys():
                          suggestion_dict[tutorial_order[0]] = [1, self.get_description(tutorial_order[0])]
                      else:
                          curr_val = suggestion_dict[tutorial_order[0]][0]
                          suggestion_dict[tutorial_order[0]][0] = curr_val + 1

        s = sum([pair[0] for pair in suggestion_dict.values()])
        for k, v in suggestion_dict.items():
            suggestion_dict[k][0] = str(round(v[0] * 100 / s, 1)) + '%'

        sorted_dict = sorted(suggestion_dict.items(), key=lambda kv: v[0], reverse=True)
        suggestions = [{'name': item[0], 'probability': item[1][0], 'description': item[1][1]}  for item in sorted_dict]

        return suggestions

    '''
    Input Parameters:   filtered_dict: list of dictionaries
                    List of codeblocks compatible with dataset
                state: JSON string
                    current state of the application
    Description:        Get next set of crowd suggestions given current state
    Output:             suggestions: list of dictionaries
                    Each dictionary is a codeblock that is crowd suggested, with the keys 'name', 'probability', and 'description'. List is ranked in descending order by probability. 
    '''
    def get_crowd_suggestions(self, dataset, state=None):
        suggestion_dict = dict()
        suggestions = []
        current_analysis = json.loads(state)
        current_id = "-1"
        checked = {}

        # do we have a name?
        if current_analysis:
          # get the id
          current_id = self.getIdForBlockName(current_analysis)
          # this recommendation was from the experts, need something from the crowd
          if current_id and current_id not in self.crowdIds:
            # get the relevant crowd ids
            candidates = self.retrieveCrowdBlockIdsByBlockTags(current_id)
            # do we have any crowd blocks with overlapping tags?
            if candidates and len(candidates) > 0:
              # randomly pick a block from the list for now
              candidate_count = 0
              while(True):
                rand_int = random.randint(0,len(candidates)-1)
                new_id = list(candidates)[rand_int]
                # get the corresponding block name
                current_analysis = self.codeblock_id_to_name(new_id)
                candidate_count += 1

                if (current_analysis in checked.keys() and checked[current_analysis]):
                  break
                else:
                  check = self.check_codeblock(current_analysis, dataset, state)
                  checked[current_analysis] = check
                  if (check):
                    # print("get_crowd_suggestions: new current_analysis",current_analysis)
                    break

                if(candidate_count == len(candidates)):
                  break


        # automated analysis needs one more step from name code block name to id
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary_code_map.json', 'r') as f_dict:
            json_dict = json.load(f_dict)

        for i in range(0, len(json_dict)):
            if current_analysis in json_dict[i]["user-data"]["method"]:
                current_id = json_dict[i]["id"]

        # print('get_crowd_suggestions, current_id: ' + str(current_id))

        for notebook_id, tutorial_order in self.json_data.items():
            for index, step in enumerate(tutorial_order):
                if current_analysis:
                    if str(step) == current_id and index < len(tutorial_order) - 1:

                        next_analysis = self.codeblock_id_to_name(str(tutorial_order[index + 1]))
                        if (next_analysis in checked.keys()):
                          check = checked[next_analysis]
                        else:
                          check = self.check_codeblock(next_analysis, dataset, state)
                          checked[next_analysis] = check

                        if next_analysis and check:
                            if next_analysis not in suggestion_dict:
                                suggestion_dict[next_analysis] = [1, self.get_description(next_analysis)]
                            else:
                                curr_val = suggestion_dict[next_analysis][0]
                                suggestion_dict[next_analysis][0] = curr_val + 1
                else:
                    next_analysis = self.codeblock_id_to_name(str(tutorial_order[0]))
                    if (next_analysis in checked.keys()):
                      check = checked[next_analysis]
                    else:
                      check = self.check_codeblock(next_analysis, dataset, state)
                      checked[next_analysis] = check

                    if next_analysis and check:
                        if next_analysis not in suggestion_dict:
                            suggestion_dict[next_analysis] = [1, self.get_description(next_analysis)]
                        else:
                            curr_val = suggestion_dict[next_analysis][0]
                            suggestion_dict[next_analysis][0] = curr_val + 1

        s = sum([pair[0] for pair in suggestion_dict.values()])
        for k, v in suggestion_dict.items():
            suggestion_dict[k][0] = str(round(v[0] * 100 / s, 1)) + '%'

        sorted_dict = sorted(suggestion_dict.items(), key=lambda kv: v[0], reverse=True)
        suggestions = [{'name': item[0], 'probability': item[1][0], 'description': item[1][1]}  for item in sorted_dict]
        # print('get_crowd_suggestions: ' + str(suggestions))

        return suggestions

    '''
    Input Parameters:   json_data: list of dictionaries
                        List of codeblocks in form of dictionary, with the keys 'name' and 'description' 
    Description:        Returns list of codeblocks
    Output:             analyses: list of dictionaries 
                        Each dictionary represents a codeblock.
    '''
    def get_analysis_list(self):
        analyses = []

        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary.json', 'r') as f:
            json_data = json.load(f)

        for i in range(0, len(json_data)):
            analyses.append({'name': json_data[i]["user-data"]["method"], 'desecription': json_data[i]["description"]})

        # print('get_analysis_list: ' + str(analyses))
        return analyses

    '''
    Input Parameters:   name: string
                        Name of codeblock that we want to return description of
    Description:        Returns a description of passed in codeblock
    Output:             description: string
                        Description of codeblock
    '''
    def get_description(self, name):
        description = ""
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary.json', 'r') as f:
            json_data = json.load(f)

        for i in range(0, len(json_data)):
            if json_data[i]['user-data']['method'] == name:
                description = json_data[i]['description']
                break

        return description

    '''
    Input Parameters:   blockId: string of integer
                      ID of the codeblock
                  blocks: list
                      List of codeblocks   
    Description:        For the given block id, get the ids of blocks with the same tags
    Output:             List of strings containing IDs                                          
    '''
    # for the given block id, get the ids of blocks with the same tags
    def retrieveCrowdBlockIdsByBlockTags(self,blockId,blocks=None):
      blockId = str(blockId)
      tags = self.getTagsForBlockId(blockId,blocks)
      return self.retrieveBlockIdsForTags(tags,self.crowdIds.values())

    '''
    Input Parameters:   blockId: string of integer
                      ID of the codeblock     
                  blocks: list
                      List of codeblocks
    Description:        For the given block ID, get the IDs of blocks with the same tags
    Output:             List of strings containing IDs
    '''  
    # for the given block id, get the ids of blocks with the same tags
    def retrieveExpertBlockIdsByBlockTags(self,blockId,blocks=None):
      blockId = str(blockId)
      tags = self.getTagsForBlockId(blockId,blocks)
      return self.retrieveBlockIdsForTags(tags,self.expertIds.values())

    '''
    Input Parameters:   blockId: string of integer
                        ID of the codeblock that we want the tags of. 
                    blocks: list
                        List of codeblocks
    Description:        For the given codeblock ID, retrieve the corresponding tags
    Output:             List of strings containing tags
    '''
    # for the given block id, retrieve the corresponding tags
    def getTagsForBlockId(self,blockId,blocks=None):
      blockId = str(blockId)
      if not blocks:
        blocks = self.openDictionary().values()
      for block in blocks:
        if blockId == block["id"]:
          return block["user-data"]["tags"]
      return None

    def check_codeblock(self, name, dataset, state):
      try:
        analysis = Analysis()
        res = analysis.execute_analysis(name, dataset, state)
      except:
        return False

      if not res['type'] == 'error':
        return True
      else:
        return False

    '''
    Input Parameters:   tags: list of strings
                        List of tags of the codeblock that we want the name of      
                    blocks: list
                        List of codeblocks
    Description:        Get the names of the codeblocks that match any of the tags
    Output:             List of strings containing codeblock names
    '''
    # get the names of the blocks that match any of the tags
    def retrieveBlockNamesForTags(self, tags,blocks=None):
      idxs = self.retrieveBlockIdsForTags(tags,blocks)
      #print "indexes:",idxs
      #print "methods:",[self.codeblock_id_to_name(idx) for idx in idxs]
      return [self.codeblock_id_to_name(idx) for idx in idxs]

    '''
    Input Parameters:   tags: list of strings
                        List of tags of the codeblock that we want the ID of
                    blocks: list
                        List of codeblocks
    Description:        Get the identifiers of the codeblocks that match any of the tags
    Output:             List of strings containing IDs
    ''' 
    # get the identifiers of the blocks that match any of the tags
    def retrieveBlockIdsForTags(self,tags,blocks=None):
      relevantIds = {}
      if not blocks:
        blocks = self.openDictionary().values()
      for tag in tags:
        for block in blocks:
          if tag in block["user-data"]["tags"]:
            relevantIds[block["id"]] = block
      #print "indexes:",relevantIds.keys()
      return relevantIds.keys()

    '''
    Input Parameters:   None
    Description:        Get all IDs associated with expert recommendations
    Output:             expertIds: dictionary
                            Dictionary containing expert IDs
    '''
    # get all ids associated with expert recommendations
    def getExpertIds(self):
      allBlocks = self.openDictionary()
      #print "total blocks:",len(allBlocks)
      expertIds = {}
      with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'manual_analysis_order.json', 'r') as f:
        expertOrderings = [o["analysis-order"] for o in json.load(f)]
        for ordering in expertOrderings:
          for idx in ordering:
            if idx in allBlocks:
              expertIds[idx] = allBlocks[idx]
      return expertIds

    '''
    Input Parameters:   None
    Description:        Get all IDs associated with crowd recommendations
    Output:             crowdIds: dictionary
                            Dictionary containing crowd IDs
    '''
    # get all ids associated with crowd recommendations
    def getCrowdIds(self):
      allBlocks = self.openDictionary()
      #print "total blocks:",len(allBlocks)
      crowdIds = {}
      with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + crowd_analysis_order, 'r') as f:
        crowdOrderings = (json.load(f)).values()
        for ordering in crowdOrderings:
          for idx in ordering:
            idx = str(idx)
            if idx in allBlocks:
              crowdIds[idx] = allBlocks[idx]
      return crowdIds
   
    '''
    Input Parameters:   filename: string
                            Name of the dictionary file
    Description:        Open the given or default dictionary file
    Output:             dmap: dictionary
                            Dictionary containing JSON from file
    '''
    # open the given or default dictionary file
    def openDictionary(self,filename=None):
      if not filename:
        filename = os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary_code_map.json'
      d = None
      with open(filename) as f:
        d = json.load(f)
      dmap = {}
      for b in d:
        dmap[b["id"]] = b
      return dmap
    
    '''
    Input Parameters:   blockName: string
                            Name of the codeblock that we want the ID for. 
                        blocks: list
                            List of codeblocks
    Description:        For the given codeblock name, return the ID of the first matching codeblock
    Output:             String of integer containing ID
    '''
    # for the given block name return the id of the first matching block
    def getIdForBlockName(self,blockName,blocks=None):
      if not blocks:
        blocks = (self.openDictionary()).values()
      for block in blocks:
        if blockName == block["user-data"]["method"]:
          return block["id"]
      return None

    '''
    Input Parameters:   block_id: string of integer
                        ID of the codeblock that we want the name of. 
    Description:        For the given codeblock ID, return the name of the first matching block
    Output:             name: string
                            Name of codeblock
    '''

    def codeblock_id_to_name(self, block_id):
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../' + 'dictionary_code_map.json', 'r') as f_dict:
            json_dict = json.load(f_dict)

        name = ""

        for i in range(0, len(json_dict)):
            if block_id == json_dict[i]["id"]:
                name = json_dict[i]["user-data"]["method"]

        return name
