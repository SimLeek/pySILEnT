import tensorflow as tf

from slam_recognition.util.energy.recovery import generate_recovery


def initialize_boosting(input_tensor, initial_multiplier=8):
    return tf.Variable(tf.ones_like(input_tensor) * initial_multiplier, dtype=tf.float32)


def get_boosting(input_tensor,  # type: tf.Tensor
                 exhaustion_tensor,  # type: tf.Variable
                 exhaustion_max=1,
                 excitation_max=1,
                 input_based_recovery=False,
                 constant_recovery=True,
                 for_visualizing=False):
    memory_biased_values = input_tensor ** exhaustion_tensor.value()
    max_pooled_memory = tf.nn.max_pool(memory_biased_values, (1, 3, 3, 1), strides=(1, 1, 1, 1), padding='SAME')

    has_fired = tf.where(tf.equal(memory_biased_values, max_pooled_memory),
                         tf.ones_like(max_pooled_memory),
                         tf.zeros_like(max_pooled_memory))

    fire_strength = has_fired * input_tensor

    exhaustion = has_fired * 255.0

    recovery = generate_recovery(fire_strength, input_based_recovery, constant_recovery)

    update_energy = exhaustion_tensor.assign(
        tf.clip_by_value((exhaustion_tensor * 255.0 - exhaustion + recovery) / 255.0, -exhaustion_max,
                         excitation_max)
    )

    if for_visualizing:
        has_fired2 = tf.image.grayscale_to_rgb(has_fired) * input_tensor
        update_color_normer = 255.0 / (exhaustion_max + excitation_max)
        update_color_centerer = (excitation_max / (exhaustion_max + excitation_max)) * 255.0
        update_energy = tf.image.grayscale_to_rgb(update_energy * update_color_normer + update_color_centerer)
        return has_fired2, update_energy
    else:
        return has_fired, update_energy
