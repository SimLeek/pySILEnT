import tensorflow as tf


def get_relative_to_indices(tensor, indices, rotation_relative=False):
    two = tf.constant(2, dtype=tf.int32)
    one = tf.constant(1, dtype=tf.int32)
    w = tf.shape(tensor)[1]
    h = tf.shape(tensor)[2]
    to_tile = tf.pad(tf.cast(tf.shape(tensor)[1:3], tf.int32), [[1, 1]])
    to_tile = tf.expand_dims(to_tile, -1)

    pad_shape = tf.cast(tf.tile(to_tile, [one, two]), tf.int32)
    padded_tensor = tf.pad(tensor, pad_shape, "CONSTANT")

    ind_pad = tf.tile(tf.constant([[4, 0]], dtype=tf.int32), [2, 1])
    no_pad = tf.tile(tf.constant([[0, 0]], dtype=tf.int32), [2, 1])
    cond_pad = tf.cond(tf.equal(tf.size(indices), 0), lambda: ind_pad, lambda: no_pad)
    padded_indices = tf.pad(indices, cond_pad)

    def get_slices(slice):
        if rotation_relative:
            raise NotImplementedError("Not sure how to do this right yet.")
            angle_rgb = self.padded_firing[slice[0]:slice[0] + 1, slice[1]:slice[1] + 1, slice[2]:slice[2] + 1,
                        :]
            angle_hue = tf.image.rgb_to_hsv(angle_rgb)[0:1, 0:1, 0:1, 0:1]
            angle = tf.squeeze(angle_hue)
            angle.set_shape([1])
            pad_rot = tf.contrib.image.transform(
                self.padded_firing,
                tf.contrib.image.angles_to_projective_transforms(
                    angle, tf.cast(tf.shape(self.padded_firing)[1], tf.float32), tf.cast(tf
                                                                                         .shape(
                        self.padded_firing)[2],
                                                                                         tf.float32)
                ))
        else:
            pad_rot = padded_tensor
        value_slice = pad_rot[tf.newaxis,
                      slice[0]:slice[0] + 1,
                      slice[1]:slice[1] + w * 2,
                      slice[2]:slice[2] + h * 2,
                      :
                      ]
        return tf.pad(value_slice,
                      [[0, 0], [slice[0], tf.cast(tf.shape(pad_rot), tf.int32)[0] - slice[0]], [0, 0], [0, 0],
                       [0, 0]], "CONSTANT")

    batch_items = tf.map_fn(fn=get_slices,
                            elems=padded_indices,
                            dtype=tf.float32)
    added_relative = tf.squeeze(tf.reduce_sum(batch_items, 0), [0])

    relativity_tensor = tf.cast(added_relative, tf.float32)

    return relativity_tensor


def get_relative_to_indices_regions(tensor, region_shape, indices, rotation_relative=False):
    two = tf.constant(2, dtype=tf.int32)
    one = tf.constant(1, dtype=tf.int32)
    w = region_shape[1]
    h = region_shape[2]
    to_tile = tf.pad(tf.cast(region_shape[1:3], tf.int32), [[1, 1]])
    to_tile = tf.expand_dims(to_tile, -1)

    pad_shape = tf.cast(tf.tile(to_tile, [one, two]), tf.int32)
    padded_tensor = tf.pad(tensor, pad_shape, "CONSTANT")

    ind_pad = tf.tile(tf.constant([[4, 0]], dtype=tf.int32), [2, 1])
    no_pad = tf.tile(tf.constant([[0, 0]], dtype=tf.int32), [2, 1])
    cond_pad = tf.cond(tf.equal(tf.size(indices), 0), lambda: ind_pad, lambda: no_pad)
    padded_indices = tf.pad(indices, cond_pad)

    def get_slices(slice):
        slice = tf.cast(slice, tf.int32)
        if rotation_relative:
            raise NotImplementedError("Not sure how to do this right yet.")
            angle_rgb = self.padded_firing[slice[0]:slice[0] + 1, slice[1]:slice[1] + 1, slice[2]:slice[2] + 1,
                        :]
            angle_hue = tf.image.rgb_to_hsv(angle_rgb)[0:1, 0:1, 0:1, 0:1]
            angle = tf.squeeze(angle_hue)
            angle.set_shape([1])
            pad_rot = tf.contrib.image.transform(
                self.padded_firing,
                tf.contrib.image.angles_to_projective_transforms(
                    angle, tf.cast(tf.shape(self.padded_firing)[1], tf.float32), tf.cast(tf
                                                                                         .shape(
                        self.padded_firing)[2],
                                                                                         tf.float32)
                ))
        else:
            pad_rot = padded_tensor
        value_slice = pad_rot[tf.newaxis,
                      slice[0]:slice[0] + 1,
                      slice[1]:slice[1] + w * 2,
                      slice[2]:slice[2] + h * 2,
                      :
                      ]
        return tf.pad(value_slice,
                      [[0, 0], [slice[0], tf.cast(tf.shape(pad_rot), tf.int32)[0] - slice[0]], [0, 0], [0, 0],
                       [0, 0]], "CONSTANT")

    batch_items = tf.map_fn(fn=get_slices,
                            elems=padded_indices,
                            dtype=tf.float32)

    relativity_tensor = tf.cast(batch_items, tf.float32)

    return relativity_tensor