'''
## FoamCase
This file defines a bunch of classes to post-process a foam case for
latter ML applications.
'''

import numpy as np
import re
import os

class FoamTimeSave:
    def __init__(self, case_path, time):
        self.case_path = case_path
        self.time = time
        self.time_str = str(time)
        self.time_path = os.path.join(case_path, self.time_str)
        self.fields = {}
        self.read_fields()
        
    def read_fields(self):
        '''
        ## Note
        We assume that the data is written as:
        internalField nonUniform List<scalar>
        nCells
        (
        value1
        value2
        ...
        )
        ;
        '''
        for item in os.listdir(self.time_path):
            item_path = os.path.join(self.time_path, item)
            if os.path.isfile(item_path): # we read the file
                handle = open(item_path, 'r')
                content = handle.read()
                handle.close()
                
                handle = open(item_path, 'r')
                if "FoamFile" in content:
                    # we have a field
                    self.fields[item] = {}
                    lines = handle.readlines()
                    for i, line in enumerate(lines):
                        if "class" in line:
                            self.fields[item]["type"] = line.split()[1].strip(";")
                        if "internalField" in line:
                            if "nonuniform" in line:
                                # this is a nonuniform field
                                self.fields[item]["nCells"] = int(lines[i+1])
                                lineSkips = i+3
                                
                                # read the data using numpy
                                if self.fields[item]["type"] == "volScalarField":
                                    
                                    self.fields[item]["data"] = np.genfromtxt(item_path, 
                                                                              skip_header=lineSkips, 
                                                                              max_rows=self.fields[item]["nCells"])
                                
                                elif self.fields[item]["type"] == "volVectorField" or self.fields[item]["type"] == "volTensorField":
                                    cleaned_lines = [re.sub(r'\(|\)', '', line) for line in lines[lineSkips:lineSkips+self.fields[item]["nCells"]]]
                                    self.fields[item]["data"] = np.array([list(map(float, line.split())) for line in cleaned_lines])
                        
                        
                            elif "uniform" in line:
                                # we only need to read the value
                        
                                if self.fields[item]["type"] == "volScalarField":
                                    line = line.replace(";","")
                                    self.fields[item]["uniformValue"] = float(line.split()[2])
                        
                                elif self.fields[item]["type"] == "volVectorField" or self.fields[item]["type"] == "volTensorField":
                                    line = re.sub(r'\(|\)', '', line).replace(";","")
                                    self.fields[item]["uniformValue"] = np.array(list(map(float, line.split()[2:])))
                handle.close()
                
    def write_fields_npy(self, output_path, names:list):
        output_path = os.path.join(output_path, self.time_str)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for name in names:
            if "data" in self.fields[name]:
                np.save(os.path.join(output_path, name), self.fields[name]["data"])
            elif "uniformValue" in self.fields[name]:
                Warning("Writing uniform value to npy file")
                np.save(os.path.join(output_path, name), self.fields[name]["uniformValue"])
            
            