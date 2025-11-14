

import os

"""
os.getcwd()

import os, sys

# Path to parent folder
parent = os.path.dirname(os.getcwd())
data_path = os.path.join(parent, "data")
data_path

# Add to path if not already present
if data_path not in sys.path:

    sys.path.append(data_path)

#
"""
os.chdir("..")


from data.GetCombinedData import getCombinedData


GeorgiaCombinedData = getCombinedData(state='GA')

GeorgiaCombinedData