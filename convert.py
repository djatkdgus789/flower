# -- coding: utf-8 --

import csv
import json
import os
import glob

from firebase import firebase
firebase_info = firebase.FirebaseApplication("https://mjucome-21a2b.firebaseio.com/flower_info", None)

firebase_shop = firebase.FirebaseApplication("https://mjucome-21a2b.firebaseio.com/shop", None)

path = os.getcwd() + '/files/*.csv'

for filename in glob.glob(path):
    csvfile = os.path.splitext(filename)[0]
    jsonfile = csvfile + '.json'

    with open(csvfile+'.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(jsonfile, 'w') as f:
    	json.dump(rows, f, ensure_ascii=False)





def infoupload():
	file = open(os.getcwd()+'/files/info.json', 'r')
	file = json.load(file,  encoding='utf-8')
	
	for i in file:
		upload = firebase_info.post('/flower_info', {'꽃이름' : i['\ufeff꽃이름'], '꽃말' : i['꽃말'], '자라는환경' : i['자라는환경']})

def shopupload():
	file = open(os.getcwd()+'/files/yongin_flowershop.json', 'r')
	file = json.load(file,  encoding='utf-8')
	
	for i in file:

		upload = firebase_shop.post('/shop', {'꽃집' : i['\ufeffflowershop'], '시도' : i['sido'], '시군구코드' : i['sigungu_code'], '시군구' : i['sigungu'], 
			'동코드' : i['dong_code'], '동' : i['dong'], '경도' : i['longitude'], '위도' : i['latitude'], '전화번호' : i['phone_num']})


infoupload()
shopupload()