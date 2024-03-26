import numpy as np
import pandas as pd
import os
from .FoamCase import FoamTimeSave

class MLDataSet:
    def __init__(self, case_path, save_path, time = 0):
        '''
        ## Description
        This class is used to create a dataset for ML applications.
        Data is read by the FoamTimeSave class, which is maintained
        by the current class.
        '''
        self.FoamData = FoamTimeSave(case_path, time)
        self.save_path = os.path.join(save_path, str(time))
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
    def geometric_downsample(self, wallmin, wallmax, 
                             xmin, xmax, ymin, ymax, zmin, zmax,
                             names=None):
        '''
        ## Description
        Extract the cells that are inside the box defined by the arguments. 
        '''
        assert ("Cx" in self.FoamData.fields.keys() and \
               "Cy" in self.FoamData.fields.keys() and \
               "Cz" in self.FoamData.fields.keys() and \
               "wallDistance" in self.FoamData.fields.keys()), \
                "The fields Cx, Cy, Cz and wallDistance must be present in the dataset."
        
        Cx = self.FoamData.fields["Cx"]["data"]
        Cy = self.FoamData.fields["Cy"]["data"]
        Cz = self.FoamData.fields["Cz"]["data"]
        d = self.FoamData.fields["wallDistance"]["data"]
        
        # get the index of the cells that are inside the box
        idx = np.argwhere((Cx >= xmin) & (Cx <= xmax) & \
                          (Cy >= ymin) & (Cy <= ymax) & \
                          (Cz >= zmin) & (Cz <= zmax) & \
                          (d >= wallmin) & (d <= wallmax)).flatten()
        
        # get the cells that are inside the box
        if names is None:
            names = []
            for item in self.FoamData.fields.keys():
                if "data" in self.FoamData.fields[item].keys():
                    names.append(item)
        
        self.geo_ds = {}
        for name in names:
            if name not in self.FoamData.fields.keys():
                Warning("The field {} is not present in the dataset. Continue...".format(name))
            
            self.geo_ds[name] = {}
            self.geo_ds[name]["data"] = self.FoamData.fields[name]["data"][idx]
            self.geo_ds[name]["nCells"] = len(idx)
            self.geo_ds[name]["type"] = self.FoamData.fields[name]["type"]
            
    def downsample_based_on_label(self, label:str, features:list, from_geo_ds = True,
                                  trivialValue = 1.0, tol = 0.05, ratio = 1.0, save = True):
        '''
        ## Description
        Downsample the dataset based on the triviality of the label. Trivial labels
        are filtered to create a balanced (1:1) dataset.
        
        `ratio`: the ratio of the trivial samples to the ordinary samples after downsampling.
        '''
        if from_geo_ds:
            assert "geo_ds" in self.__dict__, "The geometric downsampled dataset must be present."
            data = self.geo_ds
        else:
            data = self.FoamData.fields
            
        # check if the labels and the features are in the dataset
        assert label in data.keys(), "The label field is not present in the dataset."
        for feature in features:
            assert feature in data.keys(), "The feature {} is not present in the dataset.".format(feature)

        # type of the label should be volScalarField
        assert data[label]["type"] == "volScalarField",\
            "The label field should be a volScalarField, not {}.".format(data[label]["type"])
        
            
        # get the trivial indices
        trivial_indices = np.argwhere((abs(data[label]["data"] - trivialValue) < tol)).flatten()
        print(trivial_indices.shape)
        ordinary_indices = np.setdiff1d(np.arange(data[label]["nCells"]), trivial_indices)
        n_ordinary = len(ordinary_indices)
        n_trivial_to_keep = int(n_ordinary * ratio)
        
        trivial_indices_keep = np.random.choice(trivial_indices, n_trivial_to_keep, replace = False)
        indices_keep = np.concatenate((trivial_indices_keep, ordinary_indices))
        
        # downsample the dataset
        self.labelBased_ds_features = {}
        self.labelBased_ds_label = {}
        for feature in features:
            self.labelBased_ds_features[feature] = data[feature]["data"][indices_keep]
        self.labelBased_ds_label[label] = data[label]["data"][indices_keep]
            
        if save:
            # we save the pandas dataframe in .csv format
            df = pd.DataFrame(self.labelBased_ds_features)
            df.to_csv(os.path.join(self.save_path, "features.csv"), index = False)
            
            df = pd.DataFrame(self.labelBased_ds_label)
            df.to_csv(os.path.join(self.save_path, "label.csv"), index = False)
        
        
            
        
        
