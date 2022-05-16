from deep_emotion_recognition import DeepEmotionRecognizer

# initialize instancep
# inherited from emotion_recognition.EmotionRecognizer
# default parameters (LSTM: 128x2, Dense:128x2)
deeprec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
# train the model
deeprec.train()
# get the accuracy
print(deeprec.test_score())
# predict angry audio sample
prediction = deeprec.predict('data/validation/Actor_10/03-02-01-01-02-01-10_neutral.wav')
print(f"Prediction: {prediction}")
print(deeprec.predict_proba("data/emodb/wav/16a01Wb.wav"))