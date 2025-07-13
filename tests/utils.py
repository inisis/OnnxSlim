from urllib.parse import urlparse  # noqa: F401


def run_onnx(model_path, inputs):
    """
    Run an ONNX model with the given inputs and return the outputs.

    Args:
        model_path (str): Path to the ONNX model file
        inputs (dict): Dictionary of input name to numpy array

    Returns:
        dict: Dictionary of output name to numpy array
    """
    import onnxruntime as ort

    session = ort.InferenceSession(model_path)
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]

    # Filter inputs to only include those expected by the model
    filtered_inputs = {name: inputs[name] for name in input_names if name in inputs}

    # Run inference
    outputs = session.run(output_names, filtered_inputs)

    # Return outputs as a dictionary
    return {name: output for name, output in zip(output_names, outputs)}
