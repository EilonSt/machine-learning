import backprop_data

import backprop_network


#B
training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)

net = network.Network([784, 40, 10])

net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1,test_data=test_data)
#C
training_data, test_data = data.load(train_size=50000,test_size=10000)

net = network.Network([784, 40, 10])

net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1,test_data=test_data)
#D
training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)

net = backprop_network.Network([784, 30,30,30,30, 10])

net.SGD(training_data, epochs=30, mini_batch_size=10000, learning_rate=0.1, test_data=test_data)
