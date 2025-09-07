#
# from flask import Flask, request, render_template
# import numpy as np
# import tensorflow as tf
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# import joblib
#
# app = Flask(__name__)
#
# # Load the trained model
# model = load_model('final_model_federated.h5')
#
# # Health suggestions based on anxiety detection
# anxiety_suggestions = [
#     "Practice deep breathing exercises for 5 minutes several times a day",
#     "Try progressive muscle relaxation techniques",
#     "Maintain a regular sleep schedule",
#     "Limit caffeine and alcohol consumption",
#     "Consider mindfulness meditation",
#     "Stay physically active with regular exercise",
#     "Connect with supportive friends or family members",
#     "Journal about your thoughts and feelings"
# ]
#
# general_wellness = [
#     "Stay hydrated throughout the day",
#     "Eat a balanced diet rich in fruits and vegetables",
#     "Take regular breaks from screens and technology",
#     "Spend time in nature when possible",
#     "Practice gratitude by noting things you're thankful for",
#     "Maintain a consistent sleep schedule",
#     "Consider limiting news consumption if it increases stress"
# ]
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get input values from the form
#     input1 = float(request.form['input1'])
#     input2 = float(request.form['input2'])
#     input3 = float(request.form['input3'])
#     # scaler = MinMaxScaler()
#     scaler = joblib.load('scaler.save')
#     # # Prepare the input array
#     new_input = np.array([[input1, input2, input3]])
#
#
#     new_input_scaled = scaler.transform(new_input)
#
#     prediction = model.predict(new_input_scaled)
#
#     predicted_class = int(prediction[0][0] >= 0.5)
#
#     print("predicted class ")
#
#     print(predicted_class)
#
#     output = predicted_class
#
#     if output == 1:
#         result = "Anxiety Detected"
#         status = "warning"
#         message = "Based on your inputs, we've detected signs of anxiety. Consider the following suggestions:"
#         suggestions = anxiety_suggestions
#     else:
#         result = "No Anxiety Detected"
#         status = "success"
#         message = "Great! Your inputs don't indicate anxiety. Here are some general wellness tips:"
#         suggestions = general_wellness
#
#     return render_template('index.html',
#                            prediction_text=result,
#                            status=status,
#                            message=message,
#                            suggestions=suggestions)
#
#
# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, render_template
# import numpy as np
# import tensorflow as tf
# from keras.models import load_model
# import joblib

# app = Flask(__name__)

# # Load the binary classification model
# binary_model = load_model('final_model_federated.h5')

# # Load the multiclass classification model
# multiclass_model = load_model('final_model_time_series_multiclass_timeseries_2.h5')

# # Load scalers and encoders
# binary_scaler = joblib.load('scaler.save')
# multiclass_scaler = joblib.load('multiclass_timeseries_scaler.save')
# multiclass_label_encoder = joblib.load('multiclass_timeseries_label_encoder.save')

# # Suggestions
# anxiety_suggestions = [
#     "Practice deep breathing exercises for 5 minutes several times a day",
#     "Try progressive muscle relaxation techniques",
#     "Maintain a regular sleep schedule",
#     "Limit caffeine and alcohol consumption",
#     "Consider mindfulness meditation",
#     "Stay physically active with regular exercise",
#     "Connect with supportive friends or family members",
#     "Journal about your thoughts and feelings"
# ]

# general_wellness = [
#     "Stay hydrated throughout the day",
#     "Eat a balanced diet rich in fruits and vegetables",
#     "Take regular breaks from screens and technology",
#     "Spend time in nature when possible",
#     "Practice gratitude by noting things you're thankful for",
#     "Maintain a consistent sleep schedule",
#     "Consider limiting news consumption if it increases stress"
# ]

# suggestions_label_0 = [
#     "Maintain a regular sleep schedule for optimal mental health.",
#     "Engage in activities you enjoy to keep your mood uplifted.",
#     "Practice mindfulness or meditation a few minutes daily.",
#     "Exercise regularly to sustain emotional balance.",
#     "Stay connected with friends and family to nurture social bonds."
# ]

# suggestions_label_1 = [
#     "Take short breaks during work or study sessions to reduce tension.",
#     "Try guided breathing exercises or meditation to calm your nerves.",
#     "Limit caffeine and sugar intake, especially during stressful days.",
#     "Talk to a friend, counselor, or mentor about what's bothering you.",
#     "Set realistic goals and prioritize tasks to avoid overwhelm."
# ]

# suggestions_label_2 = [
#     "Seek professional support from a licensed therapist or counselor.",
#     "Try grounding techniques like the 5-4-3-2-1 method to calm yourself.",
#     "Avoid isolation — reach out to someone you trust.",
#     "Practice deep breathing: inhale for 4s, hold for 4s, exhale for 4s.",
#     "Avoid self-medicating — focus on healthy coping habits."
# ]


# @app.route('/')
# def home():
#     return render_template('index_2.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     input1 = float(request.form['input1'])
#     input2 = float(request.form['input2'])
#     input3 = float(request.form['input3'])

#     input_array = np.array([[input1, input2, input3]])
#     scaled_input = binary_scaler.transform(input_array)
#     prediction = binary_model.predict(scaled_input)
#     predicted_class = int(prediction[0][0] >= 0.5)

#     if predicted_class == 1:
#         result = "Anxiety Detected"
#         status = "warning"
#         message = "Based on your inputs, we've detected signs of anxiety. Consider the following suggestions:"
#         suggestions = anxiety_suggestions
#     else:
#         result = "No Anxiety Detected"
#         status = "success"
#         message = "Great! Your inputs don't indicate anxiety. Here are some general wellness tips:"
#         suggestions = general_wellness

#     return render_template('index_2.html',
#                            prediction_text=result,
#                            status=status,
#                            message=message,
#                            suggestions=suggestions)


# @app.route('/predict_multiclass', methods=['POST'])
# def predict_multiclass():
#     try:
#         timestep = 4
#         features = []
#         for t in range(timestep):
#             hr = float(request.form[f'input1_{t + 1}'])
#             st = float(request.form[f'input2_{t + 1}'])
#             eda = float(request.form[f'input3_{t + 1}'])
#             features.append([hr, st, eda])

#         features = np.array(features)  # shape (4, 3)
#         features_scaled = multiclass_scaler.transform(features)
#         features_scaled = features_scaled.reshape(1, timestep, 3)

#         prediction = multiclass_model.predict(features_scaled)
#         predicted_class = np.argmax(prediction, axis=1)[0]
#         label = multiclass_label_encoder.inverse_transform([predicted_class])[0]

#         if label == 0:
#             return render_template('index_2.html',
#                                    prediction_text=f"Low Anxiety Detected",
#                                    status="success",
#                                    message="You're showing a calm or relaxed state. Keep maintaining your healthy "
#                                            "habits!",
#                                    suggestions=suggestions_label_0)
#         elif label == 1:
#             return render_template('index_2.html',
#                                    prediction_text=f"Moderate Anxiety Detected",
#                                    status="success",
#                                    message="You're experiencing some signs of stress. Consider using self-care "
#                                            "techniques to manage it before it builds up",
#                                    suggestions=suggestions_label_1)
#         else:
#             return render_template('index_2.html',
#                                    prediction_text=f"High Anxiety Detected",
#                                    status="success",
#                                    message="High anxiety levels detected. It's important to take proactive steps and "
#                                            "consider seeking professional support if needed",
#                                    suggestions=suggestions_label_2)

#     except Exception as e:
#         return render_template('index_2.html',
#                                prediction_text="Error in prediction: " + str(e),
#                                status="warning",
#                                message="Ensure all inputs are filled correctly.",
#                                suggestions=[])


# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from keras.models import load_model
import joblib

app = Flask(__name__)

# Load the binary classification model (.tflite)
binary_interpreter = tf.lite.Interpreter(model_path="final_model_federated.tflite")
binary_interpreter.allocate_tensors()

# Get input and output details for the TFLite model
binary_input_details = binary_interpreter.get_input_details()
binary_output_details = binary_interpreter.get_output_details()

# Load the multiclass classification model (.h5)
multiclass_model = load_model('final_model_time_series_multiclass_timeseries_2.h5')

# Load scalers and encoders
binary_scaler = joblib.load('scaler.save')
multiclass_scaler = joblib.load('multiclass_timeseries_scaler.save')
multiclass_label_encoder = joblib.load('multiclass_timeseries_label_encoder.save')

# Suggestions
anxiety_suggestions = [
    "Practice deep breathing exercises for 5 minutes several times a day",
    "Try progressive muscle relaxation techniques",
    "Maintain a regular sleep schedule",
    "Limit caffeine and alcohol consumption",
    "Consider mindfulness meditation",
    "Stay physically active with regular exercise",
    "Connect with supportive friends or family members",
    "Journal about your thoughts and feelings"
]

general_wellness = [
    "Stay hydrated throughout the day",
    "Eat a balanced diet rich in fruits and vegetables",
    "Take regular breaks from screens and technology",
    "Spend time in nature when possible",
    "Practice gratitude by noting things you're thankful for",
    "Maintain a consistent sleep schedule",
    "Consider limiting news consumption if it increases stress"
]

suggestions_label_0 = [
    "Maintain a regular sleep schedule for optimal mental health.",
    "Engage in activities you enjoy to keep your mood uplifted.",
    "Practice mindfulness or meditation a few minutes daily.",
    "Exercise regularly to sustain emotional balance.",
    "Stay connected with friends and family to nurture social bonds."
]

suggestions_label_1 = [
    "Take short breaks during work or study sessions to reduce tension.",
    "Try guided breathing exercises or meditation to calm your nerves.",
    "Limit caffeine and sugar intake, especially during stressful days.",
    "Talk to a friend, counselor, or mentor about what's bothering you.",
    "Set realistic goals and prioritize tasks to avoid overwhelm."
]

suggestions_label_2 = [
    "Seek professional support from a licensed therapist or counselor.",
    "Try grounding techniques like the 5-4-3-2-1 method to calm yourself.",
    "Avoid isolation — reach out to someone you trust.",
    "Practice deep breathing: inhale for 4s, hold for 4s, exhale for 4s.",
    "Avoid self-medicating — focus on healthy coping habits."
]


@app.route('/')
def home():
    return render_template('index_2.html')


@app.route('/predict', methods=['POST'])
def predict():
    input1 = float(request.form['input1'])
    input2 = float(request.form['input2'])
    input3 = float(request.form['input3'])

    input_array = np.array([[input1, input2, input3]], dtype=np.float32)
    scaled_input = binary_scaler.transform(input_array).astype(np.float32)

    # Run inference with TFLite model
    binary_interpreter.set_tensor(binary_input_details[0]['index'], scaled_input)
    binary_interpreter.invoke()
    prediction = binary_interpreter.get_tensor(binary_output_details[0]['index'])

    predicted_class = int(prediction[0][0] >= 0.5)
    print("predicted_class", predicted_class)

    if predicted_class == 1:
        result = "Anxiety Detected"
        status = "warning"
        message = "Based on your inputs, we've detected signs of anxiety. Consider the following suggestions:"
        suggestions = anxiety_suggestions
    else:
        result = "No Anxiety Detected"
        status = "success"
        message = "Great! Your inputs don't indicate anxiety. Here are some general wellness tips:"
        suggestions = general_wellness

    return render_template('index_2.html',
                           prediction_text=result,
                           status=status,
                           message=message,
                           suggestions=suggestions)


@app.route('/predict_multiclass', methods=['POST'])
def predict_multiclass():
    try:
        timestep = 4
        features = []
        for t in range(timestep):
            hr = float(request.form[f'input1_{t + 1}'])
            st = float(request.form[f'input2_{t + 1}'])
            eda = float(request.form[f'input3_{t + 1}'])
            features.append([hr, st, eda])

        features = np.array(features)  # shape (4, 3)
        features_scaled = multiclass_scaler.transform(features)
        features_scaled = features_scaled.reshape(1, timestep, 3)

        prediction = multiclass_model.predict(features_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]
        label = multiclass_label_encoder.inverse_transform([predicted_class])[0]

        if label == 0:
            return render_template('index_2.html',
                                   prediction_text=f"Low Anxiety Detected",
                                   status="success",
                                   message="You're showing a calm or relaxed state. Keep maintaining your healthy "
                                           "habits!",
                                   suggestions=suggestions_label_0)
        elif label == 1:
            return render_template('index_2.html',
                                   prediction_text=f"Moderate Anxiety Detected",
                                   status="success",
                                   message="You're experiencing some signs of stress. Consider using self-care "
                                           "techniques to manage it before it builds up",
                                   suggestions=suggestions_label_1)
        else:
            return render_template('index_2.html',
                                   prediction_text=f"High Anxiety Detected",
                                   status="success",
                                   message="High anxiety levels detected. It's important to take proactive steps and "
                                           "consider seeking professional support if needed",
                                   suggestions=suggestions_label_2)

    except Exception as e:
        return render_template('index_2.html',
                               prediction_text="Error in prediction: " + str(e),
                               status="warning",
                               message="Ensure all inputs are filled correctly.",
                               suggestions=[])


if __name__ == "__main__":
    app.run(debug=True)
