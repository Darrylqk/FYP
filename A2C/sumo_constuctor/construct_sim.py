import xml.etree.ElementTree as ET
from xml.dom import minidom
import glob, os
from pprint import pprint
import random

from time import sleep


def createNetworkDict(folder = "sumo_files"):
    rootDir = os.getcwd()
    os.chdir(folder)
    netFile = glob.glob("*.net.xml")
    os.chdir("../..")

    netFilePath = os.path.join(rootDir, folder)
    netFilePath = os.path.join(netFilePath, netFile[0])
    tree = ET.parse(netFilePath)

    root = tree.getroot()

    networkDict = {}
    junctionList = []

    print("Creating network dictionary....\n\n")
    sleep(1)
    for elem in root.findall("edge"):
        junctionID = elem.get("from")
        if junctionID and junctionID not in junctionList:
            junctionList.append(junctionID)

    for junction in root.findall("junction"):
        ID = junction.get("id")
        if junction.get("type") == "dead_end" and ID in junctionList:
            ID = junction.get("id")
            junctionList.remove(ID)

    for junction in junctionList:
        networkDict[junction] = []
        for elem in root.findall("edge"):
            junctionID = elem.get("to")
            if junctionID == junction:
                edgeID = elem.get("id")
                networkDict[junction].append(edgeID)
    #pprint(networkDict)

    return networkDict
