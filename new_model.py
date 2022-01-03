
import numpy as np
import tensorflow as tf


class LinearCooperativeLearning(tf.keras.models.Model):
    def __init__(self, num_inputs: int, lambda_regularization: float):
        tf.keras.models.Model.__init__(self)
        self.num_inputs = num_inputs
        self.linear_layers = [tf.keras.layers.Dense(units=1,
                                                    kernel_regularizer=tf.keras.regularizers.l1(lambda_regularization))\
                              for _ in range(self.num_inputs)]

    def call(self, inputs):
        return [self.linear_layers[i](inputs[i]) for i in range(self.num_inputs)]


def linear_cooperative_learning_loss(base_loss, alpha_regularization):
    def loss(y_true, y_predict):
        return base_loss(y_true, y_predict)
    def contrastive_loss(y_predict)




class LinearCooperativeLearningLoss(tf.keras.losses.Loss):
    def __init__(self, alpha_regularization: float, lambda_regularization: float):
        pass


if __name__ == '__main__':
    X = np.random.normal(size=(1000, 2))
    Z = np.random.normal(size=(1000, 2))

    y = np.random.normal(size=1000)

    l_tf = LinearCooperativeLearning(num_inputs=2, lambda_regularization=0.1)




