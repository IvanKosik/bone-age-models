from keras import backend


def model_output_function(model, input_layer_names: list, output_layer_names: list):
    inputs = [model.get_layer(input_layer_name).input for input_layer_name in input_layer_names]
    outputs = [model.get_layer(output_layer_name).output for output_layer_name in output_layer_names]
    return backend.function(inputs, outputs)
