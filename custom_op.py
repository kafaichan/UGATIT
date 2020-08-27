import tensorflow as tf


class ReflectionPad2D(tf.keras.layers.Layer):
	def __init__(self, pad_size):
		super(ReflectionPad2D, self).__init__()
		self.pad_size = pad_size

	def call(self, inputs):
		return tf.pad(inputs, [[0, 0], [self.pad_size, self.pad_size], [self.pad_size, self.pad_size], [0, 0]], mode='REFLECT')


if __name__ == '__main__':
	x = tf.ones(shape=[1,256,256,3])
	model = ReflectionPad2D(3)
	print(model(x).shape)