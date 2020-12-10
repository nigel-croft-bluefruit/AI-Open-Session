from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
import math
import random
from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
import matplotlib.ticker as mtick

# you may need to: pip install tensorflow-addons:
import tensorflow_addons as tfa


class DoubleCosineDecay(LearningRateSchedule):
    """A LearningRateSchedule that uses 2 cosines to ramp up then back down."""

    def __init__(self, max_lr, total_batches, initial_lr_factor=0.02, final_lr_factor=0.1, peak_position_factor=0.5):
        """
        You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
        as the learning rate.

        Example: Create a schedule that starts at 2e-5, ramps up to 1e-4 then ramps down to 1e-5, with
        the maximum value 3/4 of the way through training:

        ```python
            learning_rate = DoubleCosineDecay(
                max_lr=1e-4,
                total_batches=num_epochs * len(train_y) // batch_size,
                initial_lr_factor=0.2,
                final_lr_factor=0.1,
                peak_position_factor=0.75)

            optimizer = Adam(learning_rate=learning_rate)

            model.compile(loss='categorical_crossentropy',
                            metrics=['accuracy'], optimizer=optimizer)
        ```

        Args:
          max_lr: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The peak learning rate.
          total_batches: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive. Calculate from number of samples, epochs & batch size.
          initial_lr_factor: A scalar `float32` or `float64` or a
            Python number.  Multiplied by the max_lr to get the starting lr.
          final_lr_factor: A scalar `float32` or `float64` or a
            Python number.  Multiplied by the max_lr to get the final lr.
          peak_position_factor: A scalar `float32` or `float64` or a
            Python number. Determines how far through training the peak occurs.

        Returns:
          A 1-arg callable learning rate schedule that takes the current optimizer
          step and outputs the decayed learning rate, a scalar `Tensor` of the same
          type as `max_lr`.
        """
        super(DoubleCosineDecay, self).__init__()
        self.max_lr, self.decay_steps, self.initial_lr_factor = max_lr, total_batches, initial_lr_factor
        self.final_lr_factor, self.peak_position_factor = final_lr_factor, peak_position_factor

    def __call__(self, step):
        with ops.name_scope_v2("DoubleCosineDecay") as name:

            max_learning_rate = ops.convert_to_tensor_v2(
                self.max_lr, name="max_learning_rate")

            dtype = max_learning_rate.dtype
            global_step = math_ops.cast(step, dtype)
            pi = math_ops.cast(3.14159, dtype)

            # calculate point when we start to decay again
            peak_step = int(self.decay_steps * self.peak_position_factor) - 1

            # calculate angles corresponding to current step for cosines in radians
            theta1 = math_ops.cast((pi * global_step / peak_step) - pi, dtype)
            theta2 = math_ops.cast(pi - (pi * (global_step - peak_step) / (self.decay_steps - peak_step - 1)), dtype)

            # initial & final learning rates:
            start = max_learning_rate * self.initial_lr_factor
            end = max_learning_rate * self.final_lr_factor

            # scaling factors for cosines:
            M1 = (max_learning_rate - start)/2.
            M2 = (end - max_learning_rate)/2.

            return tf.cond(step < peak_step,
                           lambda: math_ops.add(start, math_ops.multiply(
                                               math_ops.add(1., math_ops.cos(theta1)), M1,
                                               name=name)),
                           lambda: math_ops.add(max_learning_rate, math_ops.multiply(
                                               math_ops.add(1., math_ops.cos(theta2)), M2,
                                               name=name)))

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "total_batches": self.decay_steps,
            "initial_lr_factor": self.initial_lr_factor,
            "final_lr_factor": self.final_lr_factor,
            "peak_position_factor": self.peak_position_factor
        }

    def plot(self):
        s = self.decay_steps // 100
        if s < 1:
            s = 1

        steps = range(0, self.decay_steps, s)
        lr = [self(step) for step in steps]
        plt.plot(steps, lr)
        plt.ylabel('lr')
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.xlabel('batch no.')
        plt.show()


class AugmentedData(Sequence):
    """A Generator that augments audio spectrograms before each batch."""

    def __init__(self, x_set, y_set, batch_size, mixup_alpha=0.2, mixup=True, roll=False,
                 freq_mask=True, mask_height=8, crapify_set=None, crap_min_alpha=0.7, crap_max_beta=0.8, warp=None):
        """
        You can pass this generator directly into model.fit()
        in place of the training data.

        Example: Create a generator that just rolls the spectrograms
        by a random amount:

        ```python
            augmented_train = AugmentedData(train_x, train_y, batch_size,
                                mixup=False, roll=True, freq_mask=False)

            model.fit(augmented_train, epochs=num_epochs,\
              validation_data=(test_x, test_y), verbose=1)
        ```

        Args:
          x_set: np.array - The x dataset (spectrograms)
          y_set: np.array - The y dataset (labels)
          batch_size: Int - Number of samples to return each time
                      the learner requests data
          mixup: Boolean - Whether to perform mixup
          mixup_alpha: Float - This controls how often you get 50/50 or
                      60/40 mixups, rather than 5/95 or 95/5.
          roll: Boolean - Whether to roll spectrogram
          freq_mask: Boolean - Whether to perform frequency masking
          mask_height: Int - Maximum height of frequency mask.
                      It will vary from zero to this value.
          crapify_set: np.array - Dataset to use for
                      crapifying samples e.g. a set of "silence" recordings
          crap_min_alpha: Float - minimum alpha (transparency) of main spectro when
                      crapifying. 1.0 = always use full opacity / never attenuate
                      values < 0.25 may not be useful.
          crap_max_beta: Float - maximum opacity of crapification spectro
                      0.0 = effectively disables crapification,
                      1.0 = obliterate main spectro on some occasions
          warp: Int/float - Amount of distortion to apply along x axis. About
                      10% of width of spectrogram is a good starting value.
                      None disables warping



        Returns:
          A generator that will return a batch of x,y training data.
        """
        self.x, self.y, self.alpha = x_set, y_set, mixup_alpha
        self.mixup, self.roll, self.freq_mask = mixup, roll, freq_mask
        self.mask_height, self.crapify_set = mask_height, crapify_set
        self.crap_min_alpha, self.crap_max_beta = crap_min_alpha, crap_max_beta
        self.warp = warp
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = np.copy(self.x[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.warp is not None:
            batch_x = self._time_warp(batch_x, self.warp)

        if self.freq_mask:
            batch_x = self._frequency_mask(batch_x, self.mask_height)

        if self.roll:
            batch_x = self._roll(batch_x)

        if self.crapify_set is not None:
            batch_x = self._crapify(batch_x)

        if self.mixup:
            return self._mixup(batch_x, batch_y)
        else:
            return batch_x, batch_y

    def _roll(self, batch_x):
        max_shift_pct = 0.7
        direction = random.choice([-1, 1])
        height, width, c = batch_x.shape[1:]
        roll_by = int(width*random.random()*max_shift_pct*direction)
        return tf.roll(batch_x, roll_by, axis=2)

    def _frequency_mask(self, batch_x, num_rows):
        channel_mean = np.mean(batch_x, axis=(1, 2))[:, :, np.newaxis, np.newaxis]
        bs, y, x, c = batch_x.shape
        num_rows = tf.random.uniform(shape=[], maxval=num_rows, dtype=tf.int32)
        mask = tf.ones([bs, num_rows, x, 1]) * channel_mean
        start_row = random.randint(0, y-num_rows)
        batch_x[:, start_row:start_row+num_rows, :, :] = mask
        return batch_x

    def _crapify(self, batch_x):
        crap_idx = tf.random.uniform(shape=[len(batch_x)], maxval=len(self.crapify_set)-1, dtype=tf.int32)
        crap_spectros = self._roll(np.take(self.crapify_set, crap_idx, axis=0))

        alpha = tf.random.uniform(shape=[batch_x.shape[0], 1, 1, 1], minval=self.crap_min_alpha, maxval=1.0)
        beta = tf.random.uniform(shape=[batch_x.shape[0], 1, 1, 1], minval=0., maxval=self.crap_max_beta)

        batch_x = (batch_x * alpha) + (crap_spectros * beta)
        return batch_x

    def _mixup(self, batch_x, batch_y):
        # random sample the lambda value from beta distribution.
        lb = np.random.beta(self.alpha, self.alpha, len(batch_y))

        x_lb = lb.reshape(len(batch_y), 1, 1, 1)
        y_lb = lb.reshape(len(batch_y), 1)

        # Get alternative samples to do mixup with
        alt_idx = tf.random.uniform(shape=[len(batch_y)], maxval=len(self.y)-1, dtype=tf.int32)

        alt_x = np.take(self.x, alt_idx, 0)
        alt_y = np.take(self.y, alt_idx, 0)

        # Perform the mixup.
        x = batch_x * x_lb + alt_x * (1 - x_lb)
        y = batch_y * y_lb + alt_y * (1 - y_lb)

        return x, y

    def _time_warp(self, batch_x, W):
        (b, h, w, c) = batch_x.shape

        # set control points at each corner & copy to whole batch
        src_ctl = tf.expand_dims(tf.constant([[0., 0.], [0., w-1], [h-1, w-1], [h-1, 0.]]), 0)
        src_ctl = tf.repeat(src_ctl, repeats=b, axis=0)

        # create a random warp position for each member of batch
        warp_pos = tf.random.uniform(shape=[b, 1, 2], minval=W, maxval=(w-1 - W), dtype=tf.float32).numpy()
        # set y coordinate to bottom of spectro
        warp_pos[:, :, 0] = 0
        # add control point
        src_ctl = tf.concat([src_ctl, warp_pos], axis=1)
        # set y coordinate to top of spectro
        warp_pos[:, :, 0] = h-1
        # add control point
        src_ctl = tf.concat([src_ctl, warp_pos], axis=1)

        dest_ctl = np.array(src_ctl)
        # How far to warp. Must not coincide with corners, so keep at least 1 pixel from edge
        warp_amount = tf.random.uniform(shape=[b], minval=-W+1, maxval=W-1, dtype=tf.float32).numpy()
        # move "mid" control points along x axis by warp_amount
        dest_ctl[:, -1, 1] = dest_ctl[0, -1, 1] + warp_amount
        dest_ctl[:, -2, 1] = dest_ctl[0, -2, 1] + warp_amount

        warp, _ = tfa.image.sparse_image_warp(batch_x, src_ctl, dest_ctl)
        return warp.numpy()


def plot_hist(hist, offset=0):
    """
        Plot history graphs of loss / accuracy after training a model.

        Example:

        ```python
            hist = model.fit(augmented_train, epochs=num_epochs,\
              validation_data=(test_x, test_y), verbose=1)

            plot_hist(hist)
        ```

        Args:
          hist: The history structure returned by model.fit()
          offset: Int - the first batch to start plotting from.

        Returns:
          None.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 4))
    (ax1, ax2) = axes

    ax1.plot(hist.history['loss'][offset:], label='Training')
    ax1.plot(hist.history['val_loss'][offset:], label='Validation')
    ax1.legend(loc="upper right")
    ax1.set_title('Loss')
    ax1.set_xlabel('No. epoch')

    ax2.plot(hist.history['accuracy'][offset:], label='Training')
    ax2.plot(hist.history['val_accuracy'][offset:], label='Validation')
    ax2.legend(loc="lower right")
    ax2.set_title('Accuracy')
    ax2.set_xlabel('No. epoch')

    plt.show()


def show_spectro(spectro, cmap='magma', scale=40):
    """
        Display a spectrogram as an image

        Examples:

        ```python
            show_spectro(test_x[idx])

            show_spectro(extract_feature(TEST_FILE), cmap='gray')
        ```

        Args:
          spectro: The y,x,c channels of the spectrogram
          cmap: The colour map to convert amplitude to colours
          scale: Divisor to make spectrogram a sensible size on page
    """
    if scale < 10:
        scale = 10

    (h, w, c) = spectro.shape
    _, ax = plt.subplots(figsize=(w//scale, h//scale))

    spectro = np.fliplr(np.rot90(spectro[:, :, 0], 2))
    ax.axis('off')
    ax.imshow(spectro, cmap=cmap)


class GradCAM:
    ''' Calculate & display GradCAM heatmap '''
    def __init__(self, model, layerName=None):
        """
        Calculate & display GradCAM heatmap

        Example:

        ```python
            gc = GradCAM(model)
            gc.heatmap(spectro,expected_index)
        ```

        Args:
          model: The model to use
          layerName: [Optional] Name of conv layer to examine output of.
              By default the final conv layer is used.
        """
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = -1
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self._find_target_layer()

    def _find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def _compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def heatmap(self, spectro, classIdx):
        '''
            Display spectrogram with GradCAM heatmap overlaid & also
            display spectro / heatmap separately below.

            Args:
                spectro: The spectrogram in the form [y,x,c]
                classIdx: The index of the expected class. Alternatively,
                    you an also supply the index of the predicted class
                    (to see why it thought it was that class)
        '''
        self.classIdx = classIdx

        hm = self._compute_heatmap(spectro[np.newaxis, ...])
        hm = np.fliplr(np.rot90(hm, 2))
        xb_im_size = (spectro.shape[0], spectro.shape[1])
        spectro = np.rot90(spectro, 2)

        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(spectro[:, :, 0])
        ax.axis('off')
        ax.imshow(hm, alpha=0.6, extent=(0, *xb_im_size[::-1], 0),
                  interpolation='bilinear', cmap='gray')

        _, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes[0].axis('off')
        axes[0].imshow(spectro[:, :, 0])
        axes[1].axis('off')
        axes[1].imshow(hm, alpha=1.0, extent=(0, *xb_im_size[::-1], 0),
                       interpolation='bilinear', cmap='magma')
