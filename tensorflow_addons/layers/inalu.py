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
# ==============================================================================
"""Implements Improved Neural Arithmetic Logic Unit Layer."""

import tensorflow as tf
from typeguard import typechecked
from typing import List
from tensorflow_addons.utils import types

class NALURegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, reg_coef=0.1):
        self.reg_coef = reg_coef

    def __call__(self, var: List[tf.Variable]) -> tf.Tensor:
        return self.reg_coef * tf.reduce_mean(
            tf.math.maximum(tf.math.minimum(-var, var) + 20, 0)
        )

    def get_config(self):
        return {"reg_coef": float(self.reg_coef)}


ig = 0.0
im = 0.5
iw = 0.88

isg = 0.2
ism = 0.2
isw = 0.2

reg_rate = 0.05


@tf.keras.utils.register_keras_serializable(package="Addons")
class NALU(tf.keras.layers.Layer):
    @typechecked
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        regularizer: types.Regularizer = NALURegularizer(reg_rate),
        vector_gate: bool = True,
        w_initializer: types.Initializer = tf.random_normal_initializer(
            mean=iw, stddev=isw, seed=None
        ),
        m_initializer: types.Initializer = tf.random_normal_initializer(
            mean=im, stddev=ism, seed=None
        ),
        g_initializer: types.Initializer = tf.random_normal_initializer(
            mean=ig, stddev=isg, seed=None
        ),
        force_operation: str = None,
        *args,
        **kwargs
    ):
        super(NALU, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vector_gate = vector_gate
        self.reg_fn = regularizer
        self.force_operation = force_operation

        # action variables
        self.w_hat = self.add_weight(
            name="w",
            shape=(self.input_dim, self.output_dim),
            dtype=tf.float32,
            initializer=w_initializer,
            regularizer=self.reg_fn,
            trainable=True,
        )

        self.m_hat = self.add_weight(
            name="m",
            shape=(self.input_dim, self.output_dim),
            dtype=tf.float32,
            initializer=m_initializer,
            regularizer=self.reg_fn,
            trainable=True,
        )

        # gating varaible
        self.g = self.add_weight(
            name="g",
            shape=(self.output_dim,),
            dtype=tf.float32,
            initializer=g_initializer,
            regularizer=self.reg_fn,
            trainable=False,
        )

    def call(self, input):
        w1 = tf.math.tanh(self.w_hat) * tf.math.sigmoid(self.m_hat)
        a = tf.matmul(input, w1)
        m1 = tf.math.exp(
            tf.math.minimum(
                tf.matmul(tf.math.log(tf.math.maximum(tf.math.abs(input), 1e-7)), w1),
                20,
            )
        )
        ws = tf.math.abs(tf.reshape(w1, [-1]))
        xs = tf.concat([input] * w1.shape[1], axis=1)
        xs = tf.reshape(xs, shape=[-1, w1.shape[0] * w1.shape[1]])
        sgn = tf.sign(xs) * ws + (1 - ws)
        sgn = tf.reshape(sgn, shape=[-1, w1.shape[1], w1.shape[0]])
        ms = tf.math.reduce_prod(sgn, axis=2)
        g1 = tf.math.sigmoid(self.g)
        return g1 * a + (1 - g1) * m1 * tf.clip_by_value(ms, -1, 1)

    def get_gates_variables(self) -> List[tf.Variable]:
        return [self.g]
