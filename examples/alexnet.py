import torchvision.models as models
import pytorch2timeloop

# Define a pytorch-based neural network model, for example, a pre-defined alexnet from torchvision.
net = models.alexnet()

# Define the shape of a single input sample, in the following format:
# (# of channels, height, width)
# For example, the above alexnet will get a 224x224 RGB image:
input_shape = (3, 224, 224)

# Define the number of batches that will be used for the inference
batch_size = 1

# Define the directory names where the timeloop workload yaml files will be stored.
# The yaml files will be stored in ./workloads/alexnet/ in this example.
top_dir = 'workloads'
sub_dir = 'alexnet'

# By default, nn.Conv2d modules will be automatically converted, but nn.Linear modules will be ignored.
# If you want to convert nn.Linear, set the option to be true.
# The converter will change the description of nn.Linear into Convolution-like layer.
# (e.g., in_channel=in_features, out_channel=out_features, input_height=1, input_width=1, filter size = 1x1, stride = 1x1, padding = 0x0)
# If you want to ignore nn.Linear layers, set this option to be false. 
convert_fc = True

# Finally, in case there exists a layer that is only used during the training phase, define an identifier for a such layer. 
# For example, in torchvision.models.inception_v3, auxiliary classification layers are not used during the inference (e.g., InceptionAux).
# In this case, include a string that can serve as an identifier for such layers (e.g., 'Aux') in exception_module_names.
# But for the above alexnet, there is no necessity to define this. 
exception_module_names = []
print(net)

# Now, convert!
result = pytorch2timeloop.convert_model(net, input_shape, batch_size, sub_dir, top_dir, convert_fc, exception_module_names)
print(result)