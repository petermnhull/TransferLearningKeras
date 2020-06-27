import tensorflow as tf
from keras.models import Model
from keras.applications import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
import io

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.n_classes = 2
        self.mobilenet = MobileNet(weights = 'imagenet', include_top = False)
        self.pool = GlobalAveragePooling2D()
        self.dense1 = Dense(1024, activation = 'relu')
        self.dense2 = Dense(512, activation = 'relu')
        self.out = Dense(self.n_classes, activation = 'softmax')
        self.dropout = Dropout(0.5)

    def call(self, inputs, training = False):
        x = self.mobilenet(inputs)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dropout(x, training = training)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x

    def freeze_weights(self):
        # Freeze weights we want to keep
        for layer in self.layers[:1]:
            layer.trainable = False
        for layer in self.layers[1:]:
            layer.trainable = True
        return

    def get_summary(self):
        stream = io.StringIO()
        self.summary(print_fn = lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string
