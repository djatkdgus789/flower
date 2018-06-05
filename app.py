# -- coding: utf-8 --

from firebase import firebase
from flask import Flask, jsonify, request, render_template, Response, make_response, send_from_directory
from flask_restful import Resource, Api
import os
import json
import uuid

import predict


"""
0 : 개나리
1 : 나팔꽃
2 : 데이지
3 : 동자꽃
4 : 목화
5 : 백합
6 : 아부틸론
7 : 장미
8 : 해바라기

"""
firebase = firebase.FirebaseApplication("https://flower-87ee2.firebaseio.com", None)

app = Flask(__name__)
api = Api(app)

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/main.jpg')

class Upload(Resource):	
	def post(self):
		# byte -> image
		file = request.get_data()
		file = io.BytesIO(file)

		if file:
			result = predict.predict(file)

			if result[0] == 0:
			    label = "개나리"
			elif result[0] == 1:
			    label =  "나팔꽃"
			elif result[0] == 2:
			    label = '데이지'
			elif result[0] == 3:
			    label = '동자꽃'
			elif result[0] == 4:
			    label = '목화'
			elif result[0] == 5:
			    label = "백합"
			elif result[0] == 6:
			    label = "아부틸론"
			elif result[0] == 7:
			    label = "장미"
			elif result[0] == 8:
				label = "해바라기"

			print(label)


		upload = firebase.put('/uploads','/flower', {'label' : label, 'percent' : result[1]})

		return jsonify({'message' : 'image upload success!'})

class Load(Resource):
	def get(self):
		getPercent = firebase.get('/uploads' + '/flower', 'percent')
		getLabel = firebase.get('/uploads' + '/flower', 'label')

      	my_list = []
        my_list.append(getLabel)
        my_list.append(getPercent)

        data = {'label': my_list[0], 'percent': my_list[1]}
        json_string = json.dumps(data,ensure_ascii = False)
        response = Response(json_string,content_type="application/json; charset=utf-8")
        return response



api.add_resource(Upload,'/api/upload')
api.add_resource(Load,'/api/load')



if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0', port=3389)
