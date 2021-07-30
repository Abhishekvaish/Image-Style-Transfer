from flask import Flask , render_template , request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf	
import matplotlib.pyplot as plt
from io import BytesIO
import base64
app = Flask(__name__)
model  = tf.keras.models.load_model("complete_generator.h5")


def base64_encoded(img):
	fig = plt.figure()
	plt.imshow(img)
	plt.axis('off')
	buffer = BytesIO()
	plt.savefig(buffer, format='png',bbox_inches = 'tight',pad_inches = 0)
	buffer.seek(0)
	image_png = buffer.getvalue()
	buffer.close()
	graphic = base64.b64encode(image_png)
	graphic = graphic.decode('utf-8')
	return graphic


def predict(ip_img):
	ip_img = (ip_img - 127.5)/127.5  #(-1,1)

	op_img = model.predict(ip_img[tf.newaxis,:,:,:])[0]	
	op_img = (op_img+1)/2
	return op_img



@app.route('/',methods=['GET','POST'])
def index():
	if request.method == "GET":
		return render_template('index.html')
	elif request.method == "POST":

		f = request.files['file']

		ip_img = tf.io.decode_jpeg(f.read())
		ip_img = tf.image.resize(ip_img , (256,256))
		op_img = predict(ip_img)
		ip_img = tf.cast(ip_img , tf.uint8)

		data  = {
			"ip_uri":"data:image/png;base64,{}".format( base64_encoded(ip_img) ) , 
			"op_uri":"data:image/png;base64,{}".format( base64_encoded(op_img) )
		}

		return render_template('index.html',data = data)


if __name__ == "__main__":
	# set FLASK_APP=app.py set FLASK_ENV=development   flask run
	print("Server Is Online at port 5000")
	app.run()

