from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from flask import Flask, render_template, url_for

app = Flask(__name__)

path = 'crop_diseases_solutions.csv'
df = pd.read_csv(path)
# plant_dis = actual_class
# print(plant_dis)
# plant_dis = plant_dis.lower()
# plant = plant_dis.split()[0]
# parts = plant_dis.split()
# disease = ' '.join(parts[1:])
#
# file = 'crop_diseases_solutions.csv'
# column_name1 = 'Crop'
# column_name2 = 'Disease'
# df = pd.read_csv(file)
#
# condition = (df[column_name1].str.lower() == plant) & (df[column_name2].str.lower() == disease)
# rowindex = df[condition].index
# if not rowindex.empty:
#     solution = df.loc[rowindex, "Solution"].item()
#     print(solution)
# else:
#     print("No matching solution found.")

# Load your CNN model
model = load_model('models/CNNModel.h5')

cls_name = ['Apple Apple scab',
            'Apple Black rot',
            'Apple Cedar apple rust',
            'Apple healthy',
            'Background without leaves',
            'Blueberry healthy',
            'Cherry Powdery mildew',
            'Cherry healthy',
            'Corn Cercospora leaf spot Gray leaf spot',
            'Corn Common rust',
            'Corn Northern Leaf Blight',
            'Corn healthy',
            'Grape Black rot',
            'Grape Esca (Black Measles)',
            'Grape Leaf blight (Isariopsis Leaf Spot)',
            'Grape healthy',
            'Orange Haunglongbing (Citrus greening)',
            'Peach Bacterial spot',
            'Peach healthy',
            'Pepper, bell Bacterial spot',
            'Pepper, bell healthy',
            'Potato Early blight',
            'Potato Late blight',
            'Potato healthy',
            'Raspberry healthy',
            'Soybean healthy',
            'Squash Powdery mildew',
            'Strawberry Leaf scorch',
            'Strawberry healthy',
            'Tomato Bacterial spot',
            'Tomato Early blight',
            'Tomato Late blight',
            'Tomato Leaf Mold',
            'Tomato Septoria leaf spot',
            'Tomato Spider mites Two-spotted spider mite',
            'Tomato Target Spot',
            'Tomato Tomato Yellow Leaf Curl Virus',
            'Tomato Tomato mosaic virus',
            'Tomato healthy']


# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to between 0 and 1
    return img_array
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/index')
# def identify():
#     return render_template('identify.html')

@app.route('/')
def index():
    return render_template('detect.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Handle the uploaded image on the server side

        uploaded_image = request.files['image']
        img_path = './uploads/' + uploaded_image.filename
        uploaded_image.save(img_path)

        # Preprocess the image for your CNN model
        actual_class = request.form.get('actual_class')
        img_array = preprocess_image(img_path)

        if img_array is None:
            # Handle preprocessing error
            return jsonify({'error': 'Error processing the image'})

        # Perform predictions using your CNN model
        predictions = model.predict(img_array)  # Convert predictions to a list
        print(predictions)
        # Return the result as a JSON object
        result1 = {'model_prediction': predictions}
        result = cls_name[np.argmax(predictions[0])] #cls_name[predictions[0].index(max(predictions[0]))]

        print(result)
        # result2 = {'actual_class': actual_class}
        # result = cls_name[np.argmax(result1)]

        plant_dis = result
        print(plant_dis)
        plant_dis = plant_dis.lower()
        plant = plant_dis.split()[0]
        parts = plant_dis.split()
        disease = ' '.join(parts[1:])

        column_name1 = 'Crop'
        column_name2 = 'Disease'

        condition = (df[column_name1].str.lower() == plant) & (df[column_name2].str.lower() == disease)
        rowindex = df[condition].index
        if not rowindex.empty:
            solution1 = df.loc[rowindex, "Biological Solution"].item()
            solution2 = df.loc[rowindex, "Chemical Solution"].item()
            solution3 = df.loc[rowindex, "Cultural Solution"].item()

            print(solution1)
            print(solution2)
            print(solution3)
        else:
            print("No matching solution found.")


        # Render the detect template and pass the result variable
        return render_template('detect.html', result=result, solution1=solution1, solution2=solution2, solution3 = solution3)
    except Exception as e:
        print(f"Error in upload: {e}")
        return jsonify({'error': 'An error occurred'})


if __name__ == '__main__':
    app.run(debug=True)
