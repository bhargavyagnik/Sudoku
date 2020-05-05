from keras.models import load_model
import time

model=load_model("digit_rec_model.h5")

def no_prediction(t):
    prediction = list(model.predict(t.reshape(((1, 28, 28, 1))))[0])
    if max(prediction)>0.99:
        return prediction.index(max(prediction))
    return 0
