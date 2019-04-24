import pickle

# import the necessary packages
from keras.models import load_model

from datareader import DataReader
from generators import predict_generator

DATA_PATH_TEST = "../data/simple_dataset_test"
MODEL = "../model"
LABELS = "../labels"

image_sequences = DataReader.read_test_data(DATA_PATH_TEST)
print("[INFO] loading network and label map...")
model = load_model(MODEL)
label_to_folder = pickle.loads(open(LABELS, "rb").read())

predictions = model.predict_generator(
    generator=predict_generator(
        image_sequences,
        num_of_classes=image_sequences.shape[0]),
    steps=image_sequences.shape[0])

print(label_to_folder[predictions[0].argmax()])
