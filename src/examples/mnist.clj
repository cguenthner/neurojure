(ns examples.mnist
  "This demonstrates training a convolutional neural network on LeCun's, et al's, MNIST dataset
  (http://yann.lecun.com/exdb/mnist/). The MNIST dataset consists of images of pre-processed handwritten
  digits. The task is to predict the digit given the image."
  (:require [clojure.java.io :as io]
            [neurojure.core :as nn]
            [neurojure.senses :as senses]
            [neurojure.utils :as u]
            [ranvier.core :as r :refer [G]]
            [tensure.core :as m]))

(defn- process-mnist-images
  "Given an `input-stream` from an MNIST image data file, returns a tensor of shape
  `[image-count image-height image-width channels]`, where images are 28x28x1."
  [input-stream]
  (let [image-height 28
        image-width 28
        ; File has a 16-byte header.
        header-size 16
        pixels-per-image (* image-width image-height)
        bytes (u/gzip->bytes input-stream)
        total-pixels (- (alength bytes) header-size)
        image-count (/ total-pixels pixels-per-image)
        nd (m/zeros [total-pixels])]
    (dotimes [i total-pixels]
      (->> (+ i header-size)
           (nth bytes)
           ; Converts the unsigned byte from the file to a long, since Java byes are signed.
           (bit-and 0xff)
           (m/mset! nd i)))
    (m/reshape nd [image-count image-height image-width 1])))

(defn- process-mnist-labels
  "Given an `input-stream` from an MNIST labels data files, returns a one-hot encoded matrix of size
  `[num-images 10]`, with one class for each digit."
  [input-stream]
  (let [{:keys [encode]} (senses/make-one-hot-code (range 0 10))]
    (->> (u/gzip->bytes input-stream)
         ; 8-byte header.
         (drop 8)
         vec
         encode)))

(defn load-mnist-dataset
  []
  (print "Loading MNIST data...")
  (let [data
          {:test {:x (->> (senses/fetch-data [:mnist :test :images]
                                             "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
                          io/input-stream
                          process-mnist-images)
                  :y (->> (senses/fetch-data [:mnist :test :labels]
                                             "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
                          io/input-stream
                          process-mnist-labels)}
           :training {:x (->> (senses/fetch-data [:mnist :training :images]
                                                 "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
                              io/input-stream
                              process-mnist-images)
                      :y (->> (senses/fetch-data [:mnist :training :labels]
                                                 "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
                              io/input-stream
                              process-mnist-labels)}}]
    (println "Done loading MNIST data.")
    data))

(defn build-mnist-model
  [data]
  (nn/make-model :data data
                 :graph (G (-> :x
                               (- 128)
                               (/ 255)
                               (nn/conv2d-layer {:kernel-size [5 5]
                                                 :strides [1 1]
                                                 :output-channels 6})
                               nn/relu
                               (nn/max-pooling2d {:pool-size [2 2]
                                                  :strides [2 2]})
                               (nn/conv2d-layer {:kernel-size [5 5]
                                                 :strides [1 1]
                                                 :output-channels 16})
                               nn/relu
                               (nn/max-pooling2d {:pool-size [2 2]
                                                  :strides [2 2]})
                               (#(reshape % (make-shape (size % {:axis 0}) (size % {:axis [1 2 3]}))))
                               (nn/dense {:units 120})
                               nn/relu
                               (nn/dense {:units 84})
                               nn/relu
                               (nn/dense {:units 10})
                               nn/relu
                               (nn/softmax {:axis 1})
                               (report :y
                                       (nn/make-multiclass-accuracy-reporter {:class-axis 1})
                                       (r/make-print-logger :space))
                               (nn/cross-entropy :y)
                               (report (r/make-value-reporter "Cost") (r/make-print-logger :space))
                               (#(tensor-if :predicting
                                            (report % (constantly "") (r/make-print-logger :newline))
                                            (report % (constantly "") (r/make-print-logger))))))
                 :optimizer r/gradient-descent-optimizer
                 :optimizer-options {:learning-rate 1e-3
                                     :report [:iteration :epoch]
                                     :batch-size 192}))

(defn main- []
  (r/set-rng-seed! 0)
  (let [data (load-mnist-dataset)
        ;; Preview images from the training dataset.
        _ (->> (m/select-axis-range (get-in data [:training :x]) 0 [0 64])
               senses/data->image-array
               senses/open-image)
        naive-model (build-mnist-model data)
        trained-model (nn/train-model naive-model 10000)]
    (nn/evaluate-model trained-model :test)
    ;; => Accuracy: 99.1%
    ))
