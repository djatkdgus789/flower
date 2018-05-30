# -- coding: utf-8 --

from firebase import firebase
from flask import Flask, jsonify, request, render_template, Response, make_response, send_from_directory
from flask_restful import Resource, Api
from werkzeug import secure_filename
import os
import json
import uuid
from werkzeug import SharedDataMiddleware

import predict


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
UPLOAD_FOLDER = 'uploads'


def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS



firebase = firebase.FirebaseApplication("https://flower-87ee2.firebaseio.com", None)

app = Flask(__name__)
api = Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/main.jpg')

class Upload(Resource):	
	def post(self):
		file = request.files['file']
		if file and allowed_file(file.filename):

			# file을 전송된 폼 데이터 위조를 방지, 파일명 보호
			filename = secure_filename(file.filename)
			file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(file_path)

			filename = my_random_string(6) + filename
			os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))

			result = predict.predict(file)

			if result[0] == 0:
			    label = "안투리움"
			elif result[0] == 1:
			    label =  "Ball Moss"
			elif result[0] == 2:
			    label = '참매발톱'
			elif result[0] == 3:
			    label = '가자니아'
			elif result[0] == 4:
			    label = "장미"
			elif result[0] == 5:
			    label = "해바라기"
			elif result[0] == 6:
			    label = "wall flower"
			elif result[0] == 7:
			    label = "yellow iris"

			print(label)


		upload = firebase.put('/uploads','/flower', {'image':filename, 'label' : label, 'percent': result[1]})

		return jsonify({'message' : 'image upload success!'})

class Load(Resource):
	def get(self):
		getImage = firebase.get('/uploads' + '/flower', 'image')
		getLabel = firebase.get('/uploads' + '/flower', 'label')
		getPercent = firebase.get('/uploads' + '/flower', 'percent')

		my_list = []
		my_list.append(getLabel)
		my_list.append(getImage)
		my_list.append(getPercent)

		data = {'label': my_list[0], 'image': my_list[1], 'percent': my_list[2]}
		json_string = json.dumps(data,ensure_ascii = False)
		response = Response(json_string,content_type="application/json; charset=utf-8")
		return response

		
		
		
		

		

		


api.add_resource(Upload,'/api/upload')
api.add_resource(Load,'/api/load')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	filename = firebase.get('/uploads' + '/flower', 'image')
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)

app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})


if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0', port=3000)
