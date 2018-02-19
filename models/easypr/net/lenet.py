from .layer import *


class Lenet:
    def __init__(self):
        self.num_classes = 65

        self.x = None
        self.y = None
        self.keep_prob = None

        self.pred_logits = None
        self.pred_labels = None

        self.accuracy = None

        self.l2_loss = None

        self.weights = []
        self.biases = []

    def compile(self):

        self.keep_prob = tf.placeholder(tf.float32)

        self.weights = []
        self.biases = []

        input = ImageLayer(20, 20, 1, layer_name='LENET_IMAGE')
        label = LabelLayer()

        convpools1 = ConvPoolLayer(input, 5, 5, 16, 2, 2, layer_name='LENET_1')

        convpools2 = ConvPoolLayer(convpools1, 5, 5, 32, 2, 2, layer_name='LENET_2')

        dp = DropoutLayer(convpools2, self.keep_prob, layer_name='LENET_DP')

        flatten = FlattenLayer(dp, layer_name='LENET_FLATTEN')
        ip1 = DenseLayer(flatten, 256, layer_name="LENET_DENSE1")
        self.weights += ip1.weights
        self.biases += ip1.biases
        ip1_relu = ActivationLayer(ip1, layer_name='LENET_ACT')

        pred = OutputLayer(ip1_relu, self.num_classes, layer_name='LENET_OUTPUT')
        self.weights += pred.weights
        self.biases += pred.biases

        self.x = input.output
        self.y = label.output
        self.pred_logits = pred.output
        self.pred_labels = tf.argmax(self.pred_logits, 1)

        correct_pred = tf.equal(self.pred_labels, self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        for w in self.weights + self.biases:
            l2_loss = tf.nn.l2_loss(w)
            if self.l2_loss is None:
                self.l2_loss = l2_loss
            else:
                self.l2_loss += l2_loss
