import numpy as np
import tensorflow as tf

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def tflite_predict(interpreter, inputs):
    # Assumes inputs is a list of numpy arrays: [X_s, X_t, X_e]
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Set input tensors
    for i, arr in enumerate(inputs):
        interpreter.set_tensor(input_details[i]['index'], arr.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output
