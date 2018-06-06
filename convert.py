# -- coding: utf-8 --

import csv
import json
import os
import glob

from firebase import firebase
firebase = firebase.FirebaseApplication("https://mjucome-21a2b.firebaseio.com/flower_info", None)

path = os.getcwd() + '/files/*.csv'

for filename in glob.glob(path):
    csvfile = os.path.splitext(filename)[0]
    jsonfile = csvfile + '.json'

    with open(csvfile+'.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(jsonfile, 'w') as f:
    	json.dump(rows, f, ensure_ascii=False)





def upload():
	file = open(os.getcwd()+'/files/test.json', 'r')
	file = json.load(file,  encoding='utf-8')
	
	for i in file:
		upload = firebase.post('/flower_info', {'꽃이름' : i['\ufeff꽃이름'], '꽃말' : i['꽃말'], '자라는환경' : i['자라는환경']})


upload()