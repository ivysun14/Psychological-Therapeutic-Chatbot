#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This python code organizes the transcript data and metadata.
"""
import json
import pandas as pd
#from anyascii import anyscii

INPUT_PATH = "data/publication_metadata.csv"
OUTPUT_PATH = "processed/meta.json"
DATA_PATH = "data/transcript/"

with open(INPUT_PATH, "r") as file:
    meta = pd.read_csv(file)
    
data = meta[["Entity_ID", "file_name","Abstract","Client_Age", "Client_Gender", 
             "Client_Marital_Status", "Previous_Session_ID", "Next_Session_ID",
             "Client_Sexual_Orientation", "CTIV_category", "Psyc_Subjects", 
             "Symptoms", "Therapies", "Therapist", "Race_of_Therapist", 
             "Race_of_Client"]]

data = data.set_index('Entity_ID')

documents = list(data.index)

linked_session = data.T.to_dict(orient = "dict")

verbs = []

#Remoev the time stamps
def remove_time(lines):
    if lines.find("[") != -1: 
        pos = lines.index("[")
        #print(lines)
        try:
            end = lines.index("]")
        except:
            end = len(lines)
            
        newline = lines[:pos] + lines[end+1:]
    else:
        newline = lines
    return newline

def remove_speaker(lines):
    return lines.index(":") + 1

#Need to further fix this: verb extracted are not precise
def action_verbs(lines):
    pos = lines.find("(")
    verb = ""
    if pos != -1:
        try:
            end = lines[pos:].index(")")
        except:
            try:
                end = lines[pos:].index(" ")
            except:
                end = len(lines) - 1
                
        verb = lines[lines.index("(") : end+pos + 1]
        newline = lines[:pos] + lines[end + pos + 1:]
    else:    
        newline = lines
    
    return verb, newline

def remove_end(lines):
    try:
        newline = lines[:lines.index("</p>")]
    except:
        newline = lines
    return newline

def remove_unicode(lines):
    newline = lines.encode("ascii", "ignore")
    newline = newline.decode()
    return newline

#process the lines by adding to dic, return verbs
def process_doc(key, dic):
    print("processing " + str(keys))
    client = []
    therapist = []
    doc = open(DATA_PATH + dic[key]['file_name'], "r")
    for lines in doc.readlines():
        verb, newline = action_verbs(lines)
        newline = remove_time(newline)
        #Check who is talking
        
        if lines.startswith("<p>CLIENT:"):
            newline = newline[remove_speaker(newline) :]
            newline = remove_end(newline)
            newline = remove_unicode(newline)
            client.append(newline)
        elif lines.startswith("<p>THERAPIST:"):
            newline = newline[remove_speaker(newline) :]
            newline = remove_end(newline)
            newline = remove_unicode(newline)
            therapist.append(newline)
        
        
        if verb != "":
            verbs.append(verb)
            
        
    dic[key]['Client_Text'] = client
    dic[key]['Therapist_Text'] = therapist
    
    return verbs
    

problem = []

for keys in documents:
    
    try:
        process_doc(keys, linked_session)
    except:
        print("having problem processing: " + str(keys))
        problem.append(keys)
    
    #process_doc(keys, linked_session)        

#Write into json
with open(OUTPUT_PATH, "w") as fout:
    json.dump(linked_session, fout, indent=2)
        
    
                    
            
        
            











