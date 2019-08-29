# Neurojure

Neurojure is a Clojure library for building, training, and testing neural network models using established
architectures and optimization techniques. It uses [Ranvier](https://github.com/cguenthner/ranvier) for
optimization and [Tensure](https://github.com/cguenthner/tensure) for tensor computations.

Neurojure provides functions for implementing the following techniques:

- _Deep feedforward networks_
- _Convolutional networks_ - the `conv2d` layer works on datasets of two spatial dimensions (width x height x
  channel)
- _Recurrent networks_ - the `recurrent` and `gru` (gated recurrent unit) layers can be used to build deep
  recurrent networks
- _Initialization_ - parameters can be initialized randomly over a variety of distributions
- _Regularization_ - indvidual layers support options for regularization. Pass the final cost through the
  `regularize` layer during training to actually apply regularization. There is also a `dropout` layer.
- _Common non-linearity and loss functions_
- _Encoding utilities_ - [neurojure.senses](https://cguenthner.github.io/neurojure/docs/neurojure.senses.html)
  has some utilities for preparing datasets for use in training networks--e.g. for fetching and caching
  datasets from the web, working with image data, generating one-hot encodings, tokenizing text, using word
  embeddings, _etc._
- _Robust optimization methods_ - With [Ranvier](https://github.com/cguenthner/ranvier), Neurojure networks
  can be trained using basic gradient descent and popular varients--gradient descent with momentum, RMSProp,
  and Adam

Neurojure is currently useful for many applications, including research of novel architectures and training of
small to medium-size networks. There are plans to extend its functionality to support a wider range of
scenarios, including training of large networks and datasets in a distributed environment.

## Getting Started

Add Neurojure to your dependencies. If using leiningen, add the following to your `:dependencies` in
`project.clj`:

```
[neurojure "0.0.1"]
```

and require the `core` namespace:

```
(require '[neurojure.core :as nn])
```

## Usage

The following snippet shows how to train a neurojure model to fit the xor function:

```
(require '[neurojure.core :as nn]
         '[ranvier.core :as r :refer [G]])

(r/set-rng-seed! 0)
(def naive-model
  (nn/make-model
    ; Models can contain named datasets; `:training`, `:dev`, and `:test` are common dataset names and are
    ; the defaults used by some functions. Here we include only one dataset, `:xor`, with three inputs,
    : `:a`, `:b`, and `:y`.
    :data {:xor {:a [1 1 0 0]
                 :b [1 0 1 0]
                 :y [0 1 1 0]}}
    ; This graph represents a very simple three-neuron network with logistic non-linearities for computing
    ; the xor function.
    :graph (G (-> (join-along 1
                              (reshape :a [4 1])
                              (reshape :b [4 1]))
                  (nn/dense {:units 2})
                  nn/logistic
                  (nn/dense {:units 1})
                  ; This will report the accuracy of our results during optimization.
                  (report (reshape :y [4 1])
                          (nn/make-binary-classifier-reporter)
                          (r/make-print-logger :space))
                  nn/logistic
                  ; The `:predicting` input is set to 1 (true) by default when we run `evaluate-model` below.
                  ; During training, it is set to 0 (false). When training, we compute cost. When predicting,
                  ; we apply a threshold.
                  (#(tensor-if :predicting
                               (> % 0.5)
                               (nn/binary-cross-entropy % (reshape :y [4 1]))))
                  (report (r/make-value-reporter "Cost") (r/make-print-logger :space))))
    :optimizer-options {:learning-rate 1
                        :report [:iteration]}))

(def trained-model (nn/train-model naive-model :xor 250))
;; Prints:
;;   Accuracy: 25.0%  Cost: 2.881609  Iteration: 1
;;   Accuracy: 50.0%  Cost: 2.810322  Iteration: 2
;;   .
;;   .
;;   .
;;   Accuracy: 100.0%  Cost: 0.096633025  Iteration: 249
;;   Accuracy: 100.0%  Cost: 0.09574343  Iteration: 250

(nn/evaluate-model trained-model :xor)
;; => #Tensure
;;    [0,
;;     1.0000,
;;     1.0000,
;;     0]
```

## Examples

The [examples](https://github.com/cguenthner/neurojure/tree/master/src/examples) directory contains a couple
examples of using neurojure to solve more practical problems:

- [`mnist.clj`](https://github.com/cguenthner/neurojure/tree/master/src/examples/mnist.clj) - training a
  convolutional neural network to recognize handwritten digits
- [`book_reviews.clj`](https://github.com/cguenthner/neurojure/tree/master/src/examples/book_reviews.clj) -
  training a recurrent neural network to predict a user's rating of a book based on their written review

## Documentation

The [API docs](https://cguenthner.github.io/neurojure/docs/index.html) have the details.

## Contributing

Pull requests and ideas are welcome--please help!

## License

Copyright © 2019 Casey Guenthner

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
