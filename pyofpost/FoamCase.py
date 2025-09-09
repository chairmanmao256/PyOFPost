'''
## FoamCase
This file defines a bunch of classes to post-process a foam case for
latter ML applications.
'''

import numpy as np
import pandas as pd
import re
import os
from scipy.interpolate import griddata

# some helper functions
def point2CellInterp():
    pass


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
                
    def build_df(self):
        '''
        ## Summary
        Build a pd.DataFrame object for all the scalar field data.
        '''
        df_dict = {item: self.fields[item]["data"] for item in self.fields.keys() if self.fields[item]["type"]=="volScalarField"}
        return pd.DataFrame(df_dict)
    
class RefField:
    '''
    ## Description
    The instance of RefField class maintains the high-fidelity data for reference.
    '''
    def __init__(self, RefPath, name, onlyUx = False, readK = False):
        '''
        '''
        self.RefPath = RefPath
        self.name = name
        self.onlyUx = onlyUx
        self.readK = readK
        self.read_field()
        
    def read_field(self):
        self.fields = {}
        self.fields["xx"] = np.genfromtxt(os.path.join(self.RefPath, self.name+"_x.txt"), skip_header=1)
        self.fields["yy"] = np.genfromtxt(os.path.join(self.RefPath, self.name+"_y.txt"), skip_header=1)
        self.fields["um"] = np.genfromtxt(os.path.join(self.RefPath, self.name+"_um.txt"),skip_header=1)
        if not self.onlyUx:
            self.fields["vm"] = np.genfromtxt(os.path.join(self.RefPath, self.name+"_vm.txt"),skip_header=1) 
            if self.readK:
                self.fields["k"] = 0.5*np.genfromtxt(os.path.join(self.RefPath, self.name+"_uu.txt"),skip_header=1)\
                                + 0.5*np.genfromtxt(os.path.join(self.RefPath, self.name+"_vv.txt"),skip_header=1)\
                                + 0.5*np.genfromtxt(os.path.join(self.RefPath, self.name+"_ww.txt"),skip_header=1)
        
class FoamLineComparison:
    '''
    ## Description
    In this class, we use the FoamTimeSave class to read the data, and use 
    griddata to interploate the data to the reference line.
    
    If you don't want to read any reference data (DNS, LES, etc.) during the comparison,
    you can simply set `readRef` to `False`.
    '''
    def __init__(self, case_paths: list, times:list, names:list, RefPath:str, RefName: str,
                 lines: dict, readK = False, readRef = True):
        '''
        ## Example input:
        FoamLineComparison(["/path/to/CBFS-CND", "/path/to/CBFS-NN"], [10000, 10000], ["CND", "NN"],
                            RefPath="./path/to/LES", RefName="CBFS_13700", 
                            lines={"x0p5": 
                                  {
                                    "start": [0.5, 0.0],
                                    "end":   [0.5, 1.0]
                                   },
                                   "x1p0": {
                                       "start": [1.0, 0.0],
                                       "end":   [1.0, 1.0]
                                   }
                                   })
        '''
        assert (len(case_paths) == len(times) and len(case_paths) == len(names)),\
            "The number of case paths, times, and names must be the same."
        
        self.case_paths = case_paths
        self.times = times
        if readRef:
            self.RefPath = RefPath
        self.RefName = RefName
        self.readK = readK
        self.readRef = readRef
        
        # self.cases = {}
        # for i, name in enumerate(names):
        #     self.cases[name] = FoamTimeSave(case_paths[i], times[i])
        print("extra deleted...")
        
        self.lines = lines
        if readRef:
            self.RefCase = RefField(RefPath, RefName,readK=readK)
        self.cases = {}
        for path, time, name in zip(case_paths, times, names):
            self.cases[name] = FoamTimeSave(path, time)
        
    def extractLine(self, npCase = 100, refSkip = 5):
        '''
        ## Description
        Extract the line data based on the line definition.
        Use griddata to get the interploated line data for both cases and references
        '''
        self.lineData = {}
        for key in self.lines:
            ls = self.lines[key]["start"]
            le = self.lines[key]["end"]
            xx = np.linspace(ls[0], le[0], npCase)
            yy = np.linspace(ls[1], le[1], npCase)
            self.lineData[key] = {}
            
            k_success = True
            
            for name in self.cases.keys():
                self.lineData[key][name] = {}
                self.lineData[key][name]["xx"] = xx
                self.lineData[key][name]["yy"] = yy
                self.lineData[key][name]["u"] = griddata((self.cases[name].fields["Cx"]["data"],
                                                     self.cases[name].fields["Cy"]["data"]),
                                                     self.cases[name].fields["U"]["data"][:,0],
                                                     (xx, yy), method='linear')
                self.lineData[key][name]["v"] = griddata((self.cases[name].fields["Cx"]["data"],
                                                     self.cases[name].fields["Cy"]["data"]),
                                                     self.cases[name].fields["U"]["data"][:,1],
                                                     (xx, yy), method='linear')
                if self.readK:
                    try:
                        self.lineData[key][name]["k"] = griddata((self.cases[name].fields["Cx"]["data"],
                                                            self.cases[name].fields["Cy"]["data"]),
                                                            self.cases[name].fields["k"]["data"],
                                                            (xx, yy), method='linear')
                    except:
                        self.lineData[key][name]["k"] = None
                        print("The k field is not present in the dataset: {}".format(name))
                        k_success=False
                        
            if self.readRef:
                self.lineData[key]["Ref"] = {}
                self.lineData[key]["Ref"]["xx"] = xx[::refSkip]
                self.lineData[key]["Ref"]["yy"] = yy[::refSkip]
                self.lineData[key]["Ref"]["u"] = griddata((self.RefCase.fields["xx"],
                                                    self.RefCase.fields["yy"]),
                                                    self.RefCase.fields["um"],
                                                    (xx[::refSkip], yy[::refSkip]), method='linear')
                self.lineData[key]["Ref"]["v"] = griddata((self.RefCase.fields["xx"],
                                                    self.RefCase.fields["yy"]),
                                                    self.RefCase.fields["vm"],
                                                    (xx[::refSkip], yy[::refSkip]), method='linear')
                if self.readK and k_success:
                    self.lineData[key]["Ref"]["k"] = griddata((self.RefCase.fields["xx"],
                                                        self.RefCase.fields["yy"]),
                                                        self.RefCase.fields["k"],
                                                        (xx[::refSkip], yy[::refSkip]), method='linear')
            
class FoamDiff:
    '''
    ## Description
    In this class, we use the FoamTimeSave class to read the simulation data, and use 
    the RefField class to store the reference data. The simulation data is then interpolated
    to the reference data's grid. The difference of the reference data and the interpolated
    simulation data is then computed. 
    '''
    def __init__(self, case_paths: list, times:list, names:list, RefPath:str, RefName: str,
                 box: dict):
        '''
        ## Example input:
        FoamLineComparison(["/path/to/CBFS-CND", "/path/to/CBFS-NN"], [10000, 10000], ["CND", "NN"],
                            RefPath="./path/to/LES", RefName="CBFS_13700", 
                            lines={
                                    "xmin": 0.0,
                                    "xmax": 1.0,
                                    "ymin": 0.0,
                                    "ymax": 1.0,
                                    "zmin": 0.0,
                                    "zmax": 1.0
                                   })
        '''
        assert (len(case_paths) == len(times) and len(case_paths) == len(names)),\
            "The number of case paths, times, and names must be the same."
        
        self.case_paths = case_paths
        self.times = times
        self.RefPath = RefPath
        self.RefName = RefName
        
        # self.cases = {}
        # for i, name in enumerate(names):
        #     self.cases[name] = FoamTimeSave(case_paths[i], times[i])
        print("extra deleted...")
        
        self.box = box
        self.RefCase = RefField(RefPath, RefName,readK=False)
        self.cases = {}
        for path, time, name in zip(case_paths, times, names):
            self.cases[name] = FoamTimeSave(path, time)
            
    def __call__(self):
        self.interp()
        self.boxMask()
        self.Metrics()
            
    def interp(self):
        '''
        Interpolate the simulated data to the reference grid.
        '''
        self.diffs = {}
        for name in self.cases.keys():
            sim = self.cases[name]
            interp_x = griddata((sim.fields["Cx"]["data"], sim.fields["Cy"]["data"]), sim.fields["U"]["data"][:,0],
                              (self.RefCase.fields["xx"], self.RefCase.fields["yy"]), 
                              method = "linear", fill_value=np.nan)
            interp_y = griddata((sim.fields["Cx"]["data"], sim.fields["Cy"]["data"]), sim.fields["U"]["data"][:,1],
                              (self.RefCase.fields["xx"], self.RefCase.fields["yy"]), 
                              method = "linear", fill_value=np.nan)
            
            diff = np.vstack((interp_x - self.RefCase.fields["um"],
                              interp_y - self.RefCase.fields["vm"])).T
            
            self.diffs[name] = diff
            
    def boxMask(self):
        '''
        Generate a mask for the cells that are in the box defined by the user.
        '''
        self.mask = ((self.RefCase.fields["xx"] <= self.box["xmax"]) & 
                     (self.RefCase.fields["xx"] >= self.box["xmin"]) &
                     (self.RefCase.fields["yy"] <= self.box["ymax"]) & 
                     (self.RefCase.fields["yy"] >= self.box["ymin"]))
    
    def Metrics(self):
        '''
        Calculate the Metrics based on the diffs and the mask
        '''
        self.metrics = {k: [] for k in self.diffs.keys()}
        for name in self.diffs.keys():
            mse = (self.diffs[name]**2).sum(axis=1)
            mse = np.where(self.mask, mse, np.nan)
            self.metrics[name].append(np.nanmean(mse))
            
            mae = np.sqrt((self.diffs[name]**2).sum(axis=1))
            mae = np.where(self.mask, mae, np.nan)
            self.metrics[name].append(np.nanmean(mae))
        self.metrics = pd.DataFrame(self.metrics, index=["MSE", "MAE"])
         
         