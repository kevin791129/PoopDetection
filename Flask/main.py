########################IMPORTS###############################################
import numpy as np
import os
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import joblib 
from flask import Flask, jsonify, request, render_template, flash
from werkzeug.utils import secure_filename
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.downloader.download('vader_lexicon')

##############################################################################
###################CREATE GLOBAL VARIABLES####################################
##############################################################################
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
global loaded_model


##############################################################################
##########CREATE THE MODEL FROM PIPELINE FILE AND LOAD CHECKPOINT#############
##############################################################################
# -- Load the checkpoints we have saved
loaded_model = tf.saved_model.load('models/poop_rcnn_model')
category_index = label_map_util.create_category_index_from_labelmap('models/label_map.pbtxt')

##############################################################################
##############################PREDICTION FUNCTIONS############################
##############################################################################
def prediction(filename):
    IMAGE_PATH = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = loaded_model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 0
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=8,
                min_score_thresh=.8,
                agnostic_mode=False,
                line_thickness=10)
    detected_image_full_name = 'Result_' + filename
    #plt.subplots()
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
#    plt.show()
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'],detected_image_full_name), bbox_inches='tight')
    return detected_image_full_name

##############################################################################
#####################POOP_MAIN PAGE LOGIC#####################################
##############################################################################
@app.route('/poop', methods=['GET', 'POST'])
def predict_poop():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return render_template('poop_index.html')
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return render_template('poop_index.html')
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        processed_image = prediction(filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], processed_image)
        return render_template('prediction.html', filename = full_filename)
    
    #return render_template('index.html')
    return render_template('poop_index.html')


##############################################################################
##################CREATE NLTK SENTIMENT INTENSITY ANALYZER####################
##############################################################################
sia = SentimentIntensityAnalyzer()


##############################################################################
################################NLP PAGE LOGIC################################
##############################################################################
@app.route('/nlp', methods=['GET', 'POST'])
def predict_nlp():
    if request.method == 'POST':
        text = request.form['review']
        sentiment = sia.polarity_scores(text)
        sentiment = get_sentiment(sentiment)
        return render_template('analyze.html', sentiment = sentiment)
    return render_template('review_index.html')
    
def get_sentiment(scores):
    if scores["compound"] > 0.05:
        return "Positive"
    elif scores["compound"] < -0.05:
        return "Negative"
    return "Neutral"


##############################################################################
###########################MAIN PAGE##########################################
##############################################################################
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

##############################################################################
#######################MAIN FUNCTION##########################################
##############################################################################
if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)