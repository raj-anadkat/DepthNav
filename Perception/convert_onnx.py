import torch

# Define the input tensor shape (batch_size, channels, height, width)
input_shape = (1, 3, 256, 256)

# Load the pre-trained PyTorch model
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True)
model.eval()

# Create an input tensor of the specified shape
input_tensor = torch.randn(input_shape)

# Export the model to ONNX format
torch.onnx.export(model, 
                  input_tensor, 
                  "midas_v21_small_256.onnx", 
                  opset_version=12, 
                  input_names=["input"], 
                  output_names=["output"], 
                  dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size", 1:"height", 2:"width"}})

