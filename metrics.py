import tensorflow as tf
from tensorflow.python.keras import backend as K

class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, noc, threshold=0.5, name='macro_f1', **kwargs):
        super(MacroF1, self).__init__(name=name, **kwargs)
        self.noc = noc
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', shape=(noc,), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(noc,), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(noc,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)  # apply threshold
        y_true = tf.cast(y_true, tf.float32)
        
        #y_true = tf.where(y_true != 2, y_true, 0)  # considering '2' labels as correctly classified
        #y_pred = tf.where(y_true != 2, y_pred, 0)  # considering '2' labels as correctly classified

        self.tp.assign_add(K.sum(y_true * y_pred, axis=0))
        self.fp.assign_add(K.sum((1 - y_true) * y_pred, axis=0))
        self.fn.assign_add(K.sum(y_true * (1 - y_pred), axis=0))

    def result(self):
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        recall = self.tp / (self.tp + self.fn + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        macro_f1 = K.mean(f1)
        return macro_f1

    def reset_states(self):
        self.tp.assign(self.tp * 0)
        self.fp.assign(self.fp * 0)
        self.fn.assign(self.fn * 0)
