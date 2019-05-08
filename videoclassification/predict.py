import pickle

from keras.models import load_model

from datareader import DataReader
from generators import predict_generator

DATA_PATH = "../data/simple_data_set_test/0078"
MODEL = "model"
LABELS = "labels"

image_sequences = DataReader.read_test_data(DATA_PATH)
print("[INFO] loading network and label map...")
model = load_model(MODEL)
label_to_folder = pickle.loads(open(LABELS, "rb").read())

print(label_to_folder[
          model.predict_generator(
              generator=predict_generator(image_sequences, num_of_classes=6), steps=1
          )[0, 0, :].argmax()
      ])
