# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =====================================================epoch
import tensorflow as tf
from typeguard import typechecked
from typing import List
from tensorflow_addons.utils import types

tf.keras.backend.clear_session()


class NALURegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, reg_coef=0.1):
        self.reg_coef = reg_coef

    def __call__(self, var: List[tf.Variable]) -> tf.Tensor:
        return self.reg_coef * tf.add_n(
            [
                tf.reduce_mean(tf.math.maximum(tf.math.minimum(-v, v) + 20, 0))
                for v in var
            ]
        )

    def get_config(self):
        return {"reg_coef": float(self.reg_coef)}


# @tf.keras.utils.register_keras_serializable(package="Addons")


class NALU(tf.keras.layers.Layer):


    r"""Neural Arithmetic Logic Units
    A layer that learns addition,substraction, multiplication and division
    in transparent way. They layer has two paths: one for addition/substration
    and one for multiplication/division. We can inspect weights for these two
    paths by calling `w_hat` and `m_hat` respectively. To use this layer reliably,
    we have to delay regularization of gating varaible that switch between two paths.
    Ithave to be done by callback as from the layer-level we keep no information about epochs.
    See [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508)
    and [Improved Neural Arithmetic Logic Unit](https://arxiv.org/abs/2003.07629)
    Example:
    >>> BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE = 16, 5, 2
    >>> input = tf.random.uniform((BATCH_SIZE, INPUT_SIZE))
    >>> nalu_layer = NALU(OUTPUT_SIZE)
    >>> predict = nalu_layer(input)
    >>> assert predict.shape == (BATCH_SIZE, OUTPUT_SIZE)
    Args:
        input_dim (int): input
        output_dim (int): _description_
        regularizer (types.Regularizer, optional): _description_. Defaults to NALURegularizer(reg_coef=0.05).
        gate_as_vector (bool, optional): _description_. Defaults to True.
        clipping (float, optional): _description_. Defaults to None.
        force_operation (str, optional): _description_. Defaults to None.
        weights_separation (bool, optional): _description_. Defaults to True.
        input_gate_dependance (bool, optional): _description_. Defaults to True.
        w_initializer (types.Initializer, optional): _description_. Defaults to tf.random_normal_initializer( mean=0.88, stddev=0.2, seed=None ).
        m_initializer (types.Initializer, optional): _description_. Defaults to tf.random_normal_initializer( mean=0.5, stddev=0.2, seed=None ).
        g_initializer (types.Initializer, optional): _description_. Defaults to tf.random_normal_initializer( mean=0.0, stddev=0.2, seed=None ).
    """

    class LoopStep(tf.keras.Model):
        """It's super class to create models with NALU layer which controls the training step.
        To speed up we use @tf.function to build graph of computation, but dynamically control
        myst rely on tf.variables as all other attributes are compiled to static values.
        E.g. self.regularize controls when to start regularization and self.gating controls
        when to train gates insted of active varaivles. We can dynamically reassign values 
        to these variables to controll training steps from the callback positions (LoopControll).
        """

        def __init__(self, *args, **kwargs):
            
            super(NALU.LoopStep, self).__init__(*args, **kwargs)
            self.regularize = tf.Variable(False, trainable=False)
            self.gating = tf.Variable(False, trainable=False)
            self.gate_var = None

        def reinitialise(self):
            for l in self.layers:
                if isinstance(l, NALU):
                    l.reinitialise()

        def get_gates_variables(self) -> List[tf.Variable]:
            return [l.g for l in self.layers if isinstance(l, NALU)]

        def get_regularization_loss(self):
            return tf.math.reduce_sum(self.losses)

        @tf.function
        def train_step_active(self, data):
            if not self.gate_var:
                self.gate_var = self.get_gates_variables()
            x, y = data
            with tf.GradientTape() as tape:
                logits = self(x, training=True)
                loss_value = self.compiled_loss(y, logits)
            grads = tape.gradient(loss_value, tape.watched_variables())
            grads = [tf.clip_by_value(g, -0.1, 0.1) for g in grads]
            self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
            return {**{m.name: m.result() for m in self.metrics}, **{"ll": loss_value}}

        @tf.function
        def train_step_gating(self, data):
            if not self.gate_var:
                self.gate_var = self.get_gates_variables()
            x, y = data
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                for g in self.gate_var:
                    tape.watch(g)

                logits = self(x, training=True)
                loss_value = self.compiled_loss(y, logits)

            grads = tape.gradient(loss_value, tape.watched_variables())
            grads = [tf.clip_by_value(g, -0.1, 0.1) for g in grads]
            self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
            return {**{m.name: m.result() for m in self.metrics}, **{"ll": loss_value}}

        @tf.function
        def train_step_regularize(self):
            with tf.GradientTape(watch_accessed_variables=True) as tape:
                for g in self.gate_var:
                    tape.watch(g)
                reg_loss = self.get_regularization_loss()
            grads = tape.gradient(reg_loss, tape.watched_variables())
            grads = [tf.clip_by_value(g, -0.1, 0.1) for g in grads]
            self.optimizer.apply_gradients(zip(grads, tape.watched_variables()))
            return reg_loss


            """
            Specifying tf.function(input_signature=...) slows down the computation, but it leads to greater control:
            https://www.neuralconcept.com/post/in-graph-training-loop
            """

        @tf.function
        def train_step(self, data):
            metrics_train = tf.cond(
                self.gating,
                lambda: self.train_step_gating(data),
                lambda: self.train_step_active(data),
            )

            metrics_train["rl"] = tf.cond(
                self.regularize, lambda: self.train_step_regularize(), lambda: 0.0
            )
            return metrics_train

        # @tf.function
        def predict_loss(self, x, y):
            logits = self(x, training=False)
            return self.compiled_loss(y, logits)

            


    class LoopControl(tf.keras.callbacks.Callback):
        """Callback that controll the training loop. It controlls variables of the model that subclass LoopStep so that
        we controll how the trainign loop work still usingn default model.fit(...) functionality. We modify LoopSep-model
        variables values via accessing self.model.__controll_variables__. Using [callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback) 
        functions we controll variables values reasignment from the training-loop level. 
        Args:
            ext_data (types.TensorLike): data to check performance on extrapolation data
            ext_label (types.TensorLike): labels to check performance on extrapolation data
            int_data (types.TensorLike): data to check performance on intrapolation data
            int_label (types.TensorLike): labels to check performance on extrapolation data
            regularization_delay (int, optional): number of epochs after which regularization starts (if loss is smaller thatn regularization_loss_threshold). Defaults to 8.
            regularization_loss_threshold (float, optional): loss threshold below which regularization starts (if number of epochs is greater than regularization_delay). Defaults to 1.0.
            param_check (int, optional): number of steps after which performance on extra/inter-polation data is calculated. Results can be accessed by ext-res and int-res, respectively. Defaults to 10000.
        """

        @typechecked
        def __init__(
            self,
            ext_data: types.TensorLike,
            ext_label: types.TensorLike,
            int_data: types.TensorLike,
            int_label: types.TensorLike,
            regularization_delay: int = 8,
            regularization_loss_threshold: float = 1.0,
            param_check: int = 10000,
            *args,
            **kwargs,
        ):


            super(NALU.LoopControl, self).__init__(*args, **kwargs)
            NALU.LoopControl.regularization_delay = regularization_delay
            NALU.LoopControl.regularization_loss_threshold = regularization_loss_threshold
            NALU.LoopControl.param_check = param_check
            self.ext_data = ext_data
            self.ext_label = ext_label
            self.int_data = int_data
            self.int_label = int_label

        def on_train_begin(self, logs=None):
            NALU.LoopControl.reinit_history = [] # history after recent reinit
            NALU.LoopControl.reinit_counter = 0 # reinitialization counter
            NALU.LoopControl._steps_counter = 0 # global step counter
            NALU.LoopControl._epoch_counter = 0 # global epoch counter
            NALU.LoopControl.ext_res = [] # extrapolation data evaluation
            NALU.LoopControl.int_res = [] # interpolation data evaluation

        def on_epoch_end(self, epoch, logs=None):
            NALU.LoopControl._epoch_counter += 1

        def on_train_batch_end(self, batch, logs=None):

            # increment global step counter 
            NALU.LoopControl._steps_counter += 1

            # record last loss
            NALU.LoopControl.reinit_history.append(logs.get("loss"))

            # train either active or gating
            self.model.gating.assign(NALU.LoopControl._steps_counter % 10 > 8)

            # turn on or of ragularization depending on epoch number and last seen loss
            self.model.regularize.assign(
                NALU.LoopControl._epoch_counter >= NALU.LoopControl.regularization_delay and
                NALU.LoopControl.reinit_history[-1] < NALU.LoopControl.regularization_loss_threshold
            )

            # reinitialisation strategy
            split_index = len(NALU.LoopControl.reinit_history) // 2
            if (
                len(NALU.LoopControl.reinit_history) > 10000
                and NALU.LoopControl._epoch_counter > 0
                and NALU.LoopControl._epoch_counter % 10 == 1
                and tf.math.reduce_mean(NALU.LoopControl.reinit_history[:split_index])
                <= tf.math.add(
                    tf.math.reduce_mean(NALU.LoopControl.reinit_history[split_index:]),
                    tf.math.reduce_std(NALU.LoopControl.reinit_history[split_index:]),
                )
                and tf.math.reduce_mean(NALU.LoopControl.reinit_history[split_index:])
                > 1
            ):
                # reinitialize all nalu layers
                self.model.reinitialise()
                NALU.LoopControl.reinit_history = [] # clear recent history
                NALU.LoopControl.reinit_counter += 1 # increment 

            # check parameters
            if self._steps_counter % NALU.LoopControl.param_check == 0:
                eloss_ex = self.model.compiled_loss._losses[0](
                    self.model.predict(self.ext_data, verbose=0), self.ext_label
                )
                eloss_in = self.model.compiled_loss._losses[0](
                    self.model.predict(self.int_data, verbose=0), self.int_label
                )
                NALU.LoopControl.ext_res.append(eloss_ex)
                NALU.LoopControl.int_res.append(eloss_in)

    @typechecked
    def __init__(
        self,
        units: int,
        regularizer: types.Regularizer = NALURegularizer(reg_coef=0.05),
        clipping: float = 20,
        w_initializer: types.Initializer = tf.random_normal_initializer(
            mean=1.0, stddev=0.1, seed=None
        ),
        m_initializer: types.Initializer = tf.random_normal_initializer(
            mean=-1.0, stddev=0.1, seed=None
        ),
        g_initializer: types.Initializer = tf.random_normal_initializer(
            mean=0.0, stddev=0.1, seed=None
        ),
        *args,
        **kwargs,
    ):
        super(NALU, self).__init__(*args, **kwargs)

        self.units = units
        self.reg_fn = regularizer
        self.clipping = clipping
        
        self.w_initializer = w_initializer
        self.m_initializer = m_initializer
        self.g_initializer = g_initializer

        self.gate_as_vector = True
        self.force_operation = None
        self.weights_separation = True
        self.input_gate_dependance = False
        self.initializer = None

        

    def build(self, input_shape):
        
        # action variables
        self.w_hat = self.add_weight (
            shape = (input_shape[-1], self.units),
            initializer = self.w_initializer,
            trainable = True,
            name = "w",
            use_resource = False
        )

        self.m_hat = self.add_weight (
            shape = (input_shape[-1], self.units),
            initializer = self.m_initializer,
            trainable = True,
            name = "m",
            use_resource = False
        )

        self.w_hat_prime = self.add_weight (
            shape = (input_shape[-1], self.units),
            initializer = self.w_initializer,
            trainable = True,
            name = "w_prime",
            use_resource = False
        )

        self.m_hat_prime = self.add_weight (
            shape = (input_shape[-1], self.units),
            initializer = self.m_initializer,
            trainable = True,
            name = "m_prime",
            use_resource = False
        )

        # gating varaible
        self.g = self.add_weight (
            shape = (self.units, ),
            initializer = self.g_initializer,
            trainable = False,
            name = "g",
            use_resource = False
        )
        
    def build(self, input_shape):
        
        self.variable = self.add_weight (
            shape = (input_shape[-1], self.units),
            initializer = self.initializer,
            trainable = True,
            name = "w",
            use_resource = False
        )


    @tf.function
    def get_reg_loss(self):
        var_list = [self.w_hat, self.m_hat, self.g]
        if self.weights_separation:
            var_list += [self.w_hat_prime, self.m_hat_prime]
        return self.reg_fn(var_list)

    def call(self, input):
        eps = 1e-7
        w1 = tf.math.tanh(self.w_hat) * tf.math.sigmoid(self.m_hat)
        w2 = tf.math.tanh(self.w_hat_prime) * tf.math.sigmoid(self.m_hat_prime)
        a1 = tf.matmul(input, w1)

        m1 = tf.math.exp(tf.minimum(tf.matmul(tf.math.log(tf.maximum(tf.math.abs(input), eps)),w2), self.clipping))
        
        # sign
        w1s = tf.math.abs(tf.reshape(w2, [-1]))
        xs = tf.concat([input] * w1.shape[1], axis=1)
        xs = tf.reshape(xs, shape=[-1, w1.shape[0] * w1.shape[1]])
        sgn = tf.sign(xs) * w1s + (1 - w1s)
        sgn = tf.reshape(sgn, shape=[-1, w1.shape[1], w1.shape[0]])
        ms = tf.math.reduce_prod(sgn, axis=2)
        
        self.add_loss(lambda: self.get_reg_loss())

        g1 = tf.math.sigmoid(self.g)
        return g1 * a1 + (1 - g1) * m1 * tf.clip_by_value(ms, -1, 1)

    def reinitialise(self):
        self.g.assign(tf.random.uniform(self.g.shape, -2, 2))
        self.w_hat.assign(tf.random.uniform(self.w_hat.shape, -2, 2))
        self.m_hat.assign(tf.random.uniform(self.m_hat.shape, -2, 2))
        if self.weights_separation:
            self.w_hat_prime.assign(tf.random.uniform(self.w_hat_prime.shape, -2, 2))
            self.m_hat_prime.assign(tf.random.uniform(self.m_hat_prime.shape, -2, 2))

    def get_gates_variables(self) -> List[tf.Variable]:
        return [self.g]