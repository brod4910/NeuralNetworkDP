import numpy as np
import xml.etree.ElementTree as ET
from zipfile import ZipFile

'''
    Given either an xml file or a zip file from TSPLIB, a parsed version of 
    the xml graph will be returned as a map of vertices and their
    respective edges with costs.
'''
def prepare_data(file):
    if file[-4:] == ".zip":
        zf = ZipFile(file, "r")
        dst = "../data/{}".format(file[:-4])
        zf.extractall(dst)
        zf.close()
        file = dst

    tree = ET.parse(file)
    root = tree.getroot()

    g = root.find("graph")
    parsed_g = {}
    v_n = 0
    for v in g:
        edges = {}
        for e in v:
            edges[e.text] = float(e.attrib['cost'])
        parsed_g[v_n] = edges
        v_n += 1
            
    return parsed_g

        
if __name__ == '__main__':
    prepare_data("../data/gr17.xml")