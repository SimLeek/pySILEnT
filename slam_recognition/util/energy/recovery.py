import tensorflow as tf


def generate_constant_recovery(tensor_in, recovery_amount=4):
    return tf.ones_like(tensor_in) * recovery_amount


def generate_input_based_recovery(tensor_in, recovery_percentage=0.8):
    return tensor_in * recovery_percentage


def generate_recovery(tensor_in, is_input_based=False, is_constant=True):
    """selects which type of recovery to be used for neurons."""
    if is_input_based and not is_constant:
        recovery = generate_input_based_recovery(tensor_in)
    elif is_constant and not is_input_based:
        recovery = generate_constant_recovery(tensor_in)
    elif is_input_based and is_constant:
        recovery = tf.maximum(generate_input_based_recovery(tensor_in), generate_constant_recovery(tensor_in))
    elif not is_input_based and not is_constant:
        raise ValueError("You must choose a type of recovery")
    return recovery
