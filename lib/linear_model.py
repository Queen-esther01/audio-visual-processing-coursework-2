import os
import pickle
import numpy as np
from keras.layers import InputLayer
from keras.models import Sequential
from keras.src.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation
from keras.src.optimizers import Adam
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from  plot_bar import *
import matplotlib.pyplot as plt


class LinearModel:

    def __init__(self):
        self.loss = 'categorical_crossentropy'
        self.metrics = ['accuracy']
        self.optimizer = Adam(learning_rate=0.01)

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /yourproject/lib
        ROOT_DIR = os.path.dirname(BASE_DIR)
        base_path = os.path.join(ROOT_DIR, "results",)
        self.model_name = f"{base_path}/model_weights.weights.h5"
        self. encoder_name = f"{base_path}/label_encoder.pkl"
        self.confusion_matrix_name = f"{base_path}/confusion_matrix.png"

    def create_model(self, input_dims):
        model = Sequential()
        numClasses = 5
        # numClasses = 20
        model.add(InputLayer(shape=(input_dims,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(numClasses, activation='softmax'))

        return model

    def load_model(self, model_path, input_dim):
        model = self.create_model(input_dim)
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        model.load_weights(model_path)
        return model

    def load_label_encoder(self, label_encoder_path):
        label_encoder = pickle.load(open(label_encoder_path, 'rb'))
        return label_encoder

    def save_label_encoder(self, label_encoder_path, data):
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(data, f)

    def labelEncoder(self, labels):
        LE = LabelEncoder()
        classes = ['Jess','Josh', 'Joshua', 'Kailong', 'Ngozi']

        # classes = ['Ahmed', 'Amber', 'Charlie', 'Christopher', 'Dominic', 'Emad', 'Emma', 'Hannah', 'Imogen', 'Jess',
        #            'Josh', 'Joshua', 'Kailong', 'Kira', 'Manwel', 'Mateusz', 'Ngozi', 'Riley', 'Sivaprasath', 'Zack']
        LE = LE.fit(classes)
        transformed_label = to_categorical(LE.transform(labels))
        self.save_label_encoder(self.encoder_name, LE)

        return transformed_label

    def train_model(self, data, labels):
        X_train, X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)
        # X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
        input_dim = X_train.shape[1]
        model = self.create_model(input_dim)

        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=60, verbose=1)
        model.summary()
        model.save_weights(self.model_name)
        predicted_probabilities = model.predict(X_test, verbose=0)
        predicted = np.argmax(predicted_probabilities, axis=1)
        actual = np.argmax(y_test, axis=1)
        accuracy = metrics.accuracy_score(actual, predicted)
        print("accuracy:", accuracy * 100)
        self.predict(X_test, y_test)

        print("============")

    def predict(self, X_test, y_test):
        input_dim = X_test.shape[1]
        model = self.load_model(self.model_name,input_dim)
        LE = self.load_label_encoder(self.encoder_name)

        predicted_probabilities = model.predict(X_test, verbose=0)
        predicted = np.argmax(predicted_probabilities, axis=1)
        actual = np.argmax(y_test, axis=1)
        accuracy = metrics.accuracy_score(actual, predicted)
        print("accuracy:", accuracy * 100)

        predicted_prob = model.predict(np.expand_dims(X_test[0, :], axis=0), verbose=1)

        predicted_id = np.argmax(predicted_prob, axis=1)
        predicted_class = LE.inverse_transform(predicted_id)
        print("predicted class:", predicted_class)

        from sklearn.metrics import classification_report
        print("\nPer-class performance:")
        unique_labels = np.unique(np.concatenate([actual, predicted]))
        target_names = [LE.classes_[i] for i in unique_labels]
        print("classification_report",
              classification_report(actual, predicted, labels=unique_labels, target_names=target_names))

        confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        cm_display.plot()
        # plt.show()
        plt.savefig(self.confusion_matrix_name)
        for i in range(5):
            idx = np.random.randint(0, len(X_test))
            pred = model.predict(np.expand_dims(X_test[idx], axis=0), verbose=0)
            pred_name = LE.inverse_transform([np.argmax(pred)])[0]
            true_name = LE.inverse_transform([np.argmax(y_test[idx])])[0]
            print(f"Actual: {true_name} → Predicted: {pred_name} ({np.max(pred) * 100:.1f}%)")
        print("============")

    def top5(pred, LE):
        top_idx = np.argsort(pred[0])[::-1][:5]
        return [(LE.inverse_transform([i])[0], float(pred[0][i])) for i in top_idx]
