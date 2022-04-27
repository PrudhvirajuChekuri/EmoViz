from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, send, emit
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import cv2
import os
import shutil
from werkzeug.utils import secure_filename
from flask import current_app
from flask import send_file
from getmail import send_mail
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"
app.config["DEBUG"] = True
socketio = SocketIO(app)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
detect_fn = tf.saved_model.load("Models/FaceDetector/saved_model")#Load the face detector
model = tf.keras.models.load_model("Models/FEC")#Load the facial emotion classifier

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4'}
static_files = ['display.css', 'eye.png', 'Picdetectb.jpg', 'thumbsup.jpg', 
                'github.png', 'IU.svg', 'UI.svg', 'RT.svg', 'UV.svg', 'VU.svg', 'feedback.svg']

@app.route('/picdelete')
def picdelete():
    #When this function is called all the files that are not present in the
    #list static_files will be deleted.
    for file in os.listdir("static"):
        if file not in static_files:
            os.remove(f"static/{file}")
    return ("nothing")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/webcam')
def webcam():
    return render_template('index.html')

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def bound(boxes, scores, h, w):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 1.5)

    # define a array as matrix
    signs = []
    for i in range(len(idxs)):
            signs.append(i)
    height, width = h, w
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            ymin = int((boxes[i][0] * height))
            xmin = int((boxes[i][1] * width))
            ymax = int((boxes[i][2] * height))
            xmax = int((boxes[i][3] * width))
            signs[i] = [ymin,ymax,xmin,xmax]
    return signs

def draw_bounding_box(frame, detect_fn):
    #Returns the coordinates of the bounding boxes.
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    h, w = frame.shape[:2]
    boxes = boxes.tolist()
    scores = scores.tolist()
    coordinates = bound(boxes, scores, h, w)
    return coordinates

def detectandupdate(img):
    path = "static/" + str(img)
    image = cv2.imread(path)
    coordinates = draw_bounding_box(image, detect_fn)

    #Loop over the each bounding box.
    for (y, h, x, w) in coordinates:
        cv2.rectangle(image,(x,y),(w, h),(0, 255, 0),2)
        img2 = image[y:h, x:w]#Get the face from the image with this trick.
        img2 = tf.image.resize(img2, size = [128, 128])#Input for the model should have size-(128,128)
        pred = model.predict(tf.expand_dims(img2, axis=0))
        pred_class = class_names[tf.argmax(pred, axis = 1).numpy()[0]]
        #These conditions are just added to draw text clearly when the head is so close to the top of the image. 
        if x > 20 and y > 40:
            cv2.putText(image, pred_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, pred_class, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    path2 = f"static/pred{img}"
    #Save as predimg_name in static.
    cv2.imwrite(path2, image)


    return ([img, "pred" + img])

@app.route('/detectpic', methods=['GET', 'POST'])
def detectpic():
    UPLOAD_FOLDER = 'static'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':

        file = request.files['file']

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            result =detectandupdate(filename)
            return render_template('showdetect.html', orig=result[0], pred=result[1])

@app.route('/picdetect')
def picdetect():
    return render_template('picdetect.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/sentsafe',methods=['GET', 'POST'])
def send_sentsafe():
    if request.method == 'POST':
        email = request.form['email']
        comments = request.form['comments']
        name=request.form['name']
        comments=email+"  \n "+name+"  \n "+comments
        send_mail(email,comments)
    return render_template('sentfeed.html')

@socketio.on("message")
def handleMessage(input):
    input = input.split(",")[1]
    image_data = input
    #Since the input frame is in the form of string we need to convert it into array.
    im = Image.open(BytesIO(base64.b64decode(image_data)))
    im = np.asarray(im)
    #process it.
    coordinates = draw_bounding_box(im, detect_fn)
    for (y, h, x, w) in coordinates:
        cv2.rectangle(im,(x,y),(w, h),(0, 255, 0),2)
        img = im[y:h, x:w]
        img = tf.image.resize(img, size = [128, 128])
        pred = model.predict(tf.expand_dims(img, axis=0))
        pred_class = class_names[tf.argmax(pred, axis = 1).numpy()[0]]
        cv2.putText(im, pred_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #Convert it back into string.
    im = Image.fromarray(im)
    buff = BytesIO()
    im.save(buff, format="JPEG")
    image_data = base64.b64encode(buff.getvalue()).decode("utf-8")
    image_data = "data:image/jpeg;base64," + image_data
    emit('out-image-event', {'image_data': image_data})

def detectandupdatevideo(video):
    output_path = f"static/pred{video}"
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (640, 360))
    #using Videowriter to save the processed frames as a video.
    vidcap = cv2.VideoCapture(f"static/{video}")


    while True:

        ret, image = vidcap.read()
        if ret == True:
            coordinates = draw_bounding_box(image, detect_fn)

            for (y, h, x, w) in coordinates:
                cv2.rectangle(image,(x,y),(w, h),(0, 255, 0),2)
                img2 = image[y:h, x:w]
                img2 = tf.image.resize(img2, size = [128, 128])
                pred = model.predict(tf.expand_dims(img2, axis=0))
                pred_class = class_names[tf.argmax(pred, axis = 1).numpy()[0]]
                if x > 20 and y > 40:
                    cv2.putText(image, pred_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(image, pred_class, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            output_file = cv2.resize(image, (640, 360))
            out.write(output_file)
        else:
            break

    vidcap.release()
    out.release()

    return ([video, "pred" + video])

@app.route('/detectvideo', methods=['GET', 'POST'])
def detectvideo():
    UPLOAD_FOLDER = 'static'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':

        file = request.files['file']

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            result =detectandupdatevideo(filename)
            return render_template('showvideo.html', orig=result[0], pred=result[1])

@app.route('/video')
def video():
    return render_template('video.html')

if __name__ == "__main__":
    socketio.run(app)