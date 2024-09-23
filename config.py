from layers.fully_connected_layer import FullyConnected
from layers.flatten_layer import Flatten
from layers.conv2D import Conv2D
from layers.sigmoid import Sigmoid

model = [
    Conv2D(6,(3,3),(1,28,28)),
    Sigmoid(),
    Flatten((6,26,26),(26*26*6,1)),
    FullyConnected(26*26*6,256),
    Sigmoid(),
    FullyConnected(256,2),
    Sigmoid(),
]

epochs=50
lr = 0.1
train_and_test = False
train_only = False
test_only = True