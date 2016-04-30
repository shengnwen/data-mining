# in case you need to change the working directory, you
# can do it here. Just make sure you have unzipped the pset05 file
import os
os.chdir("../ws362_pset05")

# now  run these to load the pre-constructed files
import pset05_helper
import network

# finally, load the two datasets
mnist_train = pset05_helper.load_mnist_train()
mnist_test = pset05_helper.load_mnist_test()
# cifar_train = pset05_helper.load_cifar_train()
# cifar_test = pset05_helper.load_cifar_test()


########## Example 1:
# here is an example of constructing a neural network on MNIST with
# two hidden layers, one with 30 nodes and another with 15. The cost
# function here is the quadratic cost
# net = network.Network([784, 30, 15, 10], cost=network.QuadraticCost)
#
# # now use this to initialize the weights:
# net.large_weight_initializer()
#
# # finally, this will actually fit the weights and biases using 5 epochs,
# # a mini-batch size of 5, a learning rate of 0.5, no regularization,
# # and evaluating the accuracy on the test dataset
# net.SGD(training_data=mnist_train,
#         epochs=5,
#         mini_batch_size=5,
#         eta=0.5,
#         lmbda=0,
#         evaluation_data=mnist_test,
#         monitor_evaluation_accuracy=True)
#
# # once you have trained the model, you can print a confusion matrix between
# # the categories
# net.confusion(mnist_test)


########## Example 2:
# here is an example of constructing a neural network on MNIST with
# two hidden layers, one with 30 nodes and another with 15. The cost
# function is now cross entropy
net = network.Network([784, 80, 10, 10], cost=network.QuadraticCost)

# now use a more intelligent weighting scheme as 1/sqrt(# weights):
net.default_weight_initializer()

# finally, this will actually fit the weights and biases using 5 epochs,
# a mini-batch size of 10, a learning rate of 3, regularization of 1,
# and evaluating the accuracy on the test dataset
net.SGD(training_data=mnist_train,
        epochs=40,
        mini_batch_size=5,
        eta=0.2,
        lmbda=0,
        evaluation_data=mnist_test,
        monitor_evaluation_accuracy=True)

# once you have trained the model, you can print a confusion matrix between
# the categories
net.confusion(mnist_test)


# ########## Example 3:
# # here is an example of constructing a neural network on CIFAR-10 with
# # two hidden layers of 50 and 10 nodes each, and cross entropy loss
# net = network.Network([3072, 50, 15, 10], cost=network.CrossEntropyCost)
#
# # use a more intelligent weighting scheme as 1/sqrt(# weights):
# net.default_weight_initializer()
#
# # finally, this will actually fit the weights and biases using 5 epochs,
# # a mini-batch size of 10, a learning rate of 3, no regularization,
# # and evaluating the accuracy on the test dataset
# net.SGD(training_data=cifar_train,
#         epochs=20,
#         mini_batch_size=5,
#         eta=0.01,
#         lmbda=0,
#         evaluation_data=cifar_test,
#         monitor_evaluation_accuracy=True)
#
# # once you have trained the model, you can print a confusion matrix between
# # the categories
# net.confusion(cifar_test)

