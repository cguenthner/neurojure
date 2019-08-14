(ns examples.book-reviews
  "This example uses Amazon book review and rating data from `http://jmcauley.ucsd.edu/data/amazon/`.
  It encodes reviews using embeddings (GloVes), and trains a recurrent neural network model to predict
  the corresponding rating. Ratings are on a discrete 1 - 5 scale, but the model predicts over a continuous
  range."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [clojure.string :as string]
            [neurojure.core :as nn]
            [neurojure.senses :as senses]
            [neurojure.utils :as u :refer [def-]]
            [ranvier.core :as r :refer [G]]
            [tensure.core :as m]))

(def dataset-config
  "We include a review in the dataset if the total number of characters in it is in
  [`:min-char-count`, `max-char-count`] and if the total number of tokens in it is in [`:min-token-count`,
  `:max-token-count`]. We include only the first `:max-review-count` reviews; of these, `:test-fraction`
  are included in the test set. `:vocab-size` and `:word-vec-size` specify which GloVe set to use for
  encoding the reviews."
  {:min-char-count 3
   :max-char-count 500
   :min-token-count 1
   :max-token-count 125
   :max-review-count 2.5e5
   :test-fraction 0.01
   :vocab-size 400
   :word-vec-size 100})

(defn- load-dataset
  "Fetches the dataset if necessary (caching it locally), filters samples based on criteria in
  `dataset-config`, and returns a seq of objects like `{:rating :review :token-count}`, where `:review` is a
  seq of token strings."
  [dataset-config]
  (let [{:keys [min-char-count max-char-count
                min-token-count max-token-count
                max-review-count]} dataset-config
        file (senses/fetch-data :book-reviews
                                "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz")]
    (with-open [in (-> file
                       io/input-stream
                       java.util.zip.GZIPInputStream.
                       java.io.InputStreamReader.
                       java.io.BufferedReader.)]
      (->> (line-seq in)
           u/unchunk
           (pmap (fn [line]
                   (let [{review "reviewText"
                          rating "overall"} (json/read-str line)
                         review-len (count review)]
                     (when (and (>= review-len min-char-count)
                                (<= review-len max-char-count))
                       (let [review-tokens (senses/tokenize review)
                             token-count (count review-tokens)]
                         (when (and (>= token-count min-token-count)
                                    (<= token-count max-token-count))
                           {:rating rating
                            :review (vec review-tokens)
                            :token-count token-count}))))))
           (keep identity)
           (take max-review-count)
           vec))))

(defn- build-review-tensor
  "Given a seq of data of size `data-count` (each element like {:rating :review :token-count}), a maximum
  token count, and a map of word strings to vectors, returns an object like:
  ```
    {:x Tensor of shape [example-count word-vec-size max-token-count]
     :mask Tensor of shape [example-count max-token-count]}

  ```
  `:x` has trailing zeros along the token axis when the number of tokens in the example is less than
  `max-token-count`, and it has zeros where the corresponding token does not exist in `word-vecs`. If example
  m is of length n, then row m in `:mask` has 1's in range [m, n) and 0's in range [n max-token-count)."
  [data max-token-count word-vecs]
  (let [data (vec data)
        example-count (count data)
        vector-size (-> word-vecs vals first (m/dimension-count 0))
        reviews (m/zeros [example-count vector-size max-token-count])
        mask (m/zeros [example-count max-token-count])]
    (dotimes [i example-count]
      (when (zero? (mod i 10000))
        (println "Processing sample " i " of " example-count))
      (let [{:keys [review token-count]} (nth data i)]
        (dotimes [j token-count]
          (when-let [v (get word-vecs
                            (nth review j))]
            (m/set-range! reviews i :all j v)))
        (m/set-range! mask i [0 token-count] (m/ones [token-count]))))
    {:x reviews
     :mask mask}))

(defn- build-prediction-dataset
  "Given a seq of unprocessed reviews as strings and a `dataset-config`, returns an object like `{:x :mask}`
  (such as that produced by `build-review-tensor`)."
  [reviews dataset-config]
  (let [{:keys [max-token-count vocab-size word-vec-size]} dataset-config
        word-vecs (senses/load-gloves vocab-size word-vec-size)]
    (-> (map (fn [review]
               (let [tokens (senses/tokenize review)
                     token-count (count tokens)]
                 (when (> token-count max-token-count)
                   (u/throw-str "Cannot make prediction. Review is too long."))
                 {:review tokens
                  :token-count token-count}))
             reviews)
        (build-review-tensor max-token-count word-vecs))))

(defn- process-data
  "Given a seq of data like `{:rating :review :token-count}` and a `dataset-config`, returns a map like
  `{:training :testing}`, with values that are maps like `{:x :y :mask}`. "
  [data dataset-config]
  (let [{:keys [test-fraction vocab-size word-vec-size
                max-token-count]} dataset-config
        example-count (count data)
        test-example-count (Math/floor (* example-count test-fraction))
        shuffled-data (u/deterministic-shuffle 0 data)
        test-data (take test-example-count data)
        training-data (drop test-example-count data)
        word-vecs (senses/load-gloves vocab-size word-vec-size)
        build-dataset (fn [data]
                        (assoc (build-review-tensor data max-token-count word-vecs)
                          :y (m/reshape (m/array (map :rating data))
                                        [(count data) 1])))]
    {:training (build-dataset training-data)
     :test (build-dataset test-data)}))

(defn- predict
  "Givne a seq of raw review strings, a `dataset-config`, and a trained model, returns a tensor of shape
  `[num-reviews 1]`, wehre each element is a predicted rating corresponding to the review."
  [reviews dataset-config model]
  (->> (build-prediction-dataset reviews dataset-config)
       (nn/evaluate-model model)))

(defn- test-model
  "Given a model and a dataset like {:x :y :mask}, breaks the dataset into batches, and accumulates
  predictions. Prints a 5 x 5 matrix where rows correspond to predicted ratings, columns correspond to actual
  ratings, and cells are the count of samples with the given predicted and actual ratings."
  [model data]
  (let [actual (:y data)
        predicted (->> (range 0 2500 500)
                       (map (fn [start-index]
                              (->> (u/update-vals
                                     data
                                     #(m/select-axis-range % 0 [start-index (+ start-index 500)]))
                                   (nn/evaluate-model model))))
                       (apply m/join-along 0))]
    (->> (u/zip (m/eseq predicted) (m/eseq actual))
         (reduce (fn [result [predicted actual]]
                   (let [predicted-index (-> (int predicted)
                                             (max 1)
                                             (min 5)
                                             dec)
                         actual-index (dec (int actual))]
                     (update-in result [predicted-index actual-index] inc)))
                 (vec (repeatedly 5 #(vec (repeat 5 0)))))
         (cons (vec (range 1 6)))
         (map-indexed (fn [i row]
                        (->> (map #(format "%4d" %) row)
                             (string/join " ")
                             (#(if (zero? i)
                                 (str " P\\A" %)
                                 (str i " | " %))))))
         (string/join "\n")
         (str "Predicted\\Actual:\n")
         println)))

(defn- build-model
  [data]
  (let [batch-size 192
        accuracy-threshold 0.5
        ; Reports the percentage of predictions that are within +/- `accuracy-threshold` of the actual value.
        accuracy-reporter (fn [diff]
                            (r/make-datapoint
                              (-> (m/lt (m/abs diff)
                                        (m/array accuracy-threshold))
                                  m/esum
                                  m/->int
                                  (/ (m/row-count diff))
                                  (* 100)
                                  float)
                              :accuracy
                              :percent))
        ; Produces a seq of tuples like `[rating count]` corresponding to the distribution of predicted
        ; ratings. This is useful to monitor during training, since many models underfit the data (e.g.
        ; generating distributions of all 5's, since the majority of ratings are 5's).
        distribution-reporter (fn [predictions]
                                (r/make-datapoint
                                  (->> (m/eseq predictions)
                                       (map #(Math/round %))
                                       frequencies
                                       (sort-by first)
                                       str)
                                  :dist
                                  :string))
        ; Produces a matrix of shape `[num-examples 1]`.
        predictions (G (-> :x
                           (nn/gru {:hidden-units 50
                                    :output-units 50
                                    :recurrent-activation nn/relu
                                    :mask :mask})
                           nn/relu
                           (nn/gru {:hidden-units 20
                                    :output-units 20
                                    :recurrent-activation nn/relu
                                    :mask :mask})
                           nn/relu
                           (nn/gru {:hidden-units 5
                                    :output-units 5
                                    :recurrent-activation nn/relu
                                    :mask :mask})
                           (select-range [:all :all :last])
                           (nn/dense {:units 1})
                           nn/relu))
        g (G (tensor-if :predicting
                        ; We clip predictions for predicting but not For training.
                        (-> predictions
                            (max 1)
                            (min 5))
                        (-> predictions
                            (report distribution-reporter (r/make-print-logger :space))
                            (- :y)
                            (report accuracy-reporter (r/make-print-logger :space))
                            ; MSE
                            (pow 2)
                            esum
                            (/ (size :y {:axis 0}))
                            (report (r/make-value-reporter "Cost") (r/make-print-logger :space)))))]
    (nn/make-model :data data
                   :graph g
                   :optimizer r/gradient-descent-optimizer
                   :optimizer-options {:report [:iteration :epoch]
                                       :friction 0.9
                                       :rms-friction 0.999
                                       :learning-rate 1e-3
                                       :batch-size batch-size})))

(defn main-
  []
  (r/set-rng-seed! 0)
  (let [data (let [_ (println "Loading data...")
                   data (load-dataset dataset-config)
                   _ (println "Processing data...")
                   processed-data (process-data data dataset-config)]
               processed-data)
        iterations 1e4
        naive-model (build-model data)
        trained-model (nn/train-model naive-model iterations)]
    (test-model trained-model (nn/get-model-dataset trained-model :test))
    ;; => Predicted\Actual:
    ;;     P\A   1    2    3    4    5
    ;;    1 |   32   18    2    0    0
    ;;    2 |   27   58   40    7    8
    ;;    3 |    5   25   75   77   90
    ;;    4 |    8    7   32  359 1524
    ;;    5 |    0    0    1    3  102
    (predict ["This is a terrible book; I mean, really awful. It's hard to imagine a book worse than this."
              ; => 1.1514
              "The worst book I've ever read. Don't waste your money on this."
              ; => 1.0601
              "It's okay."
              ; => 2.9006
              "It's terrible."
              ; => 1.6036
              "It's not terrible."
              ; => 3.3664
              "It's great."
              ; => 4.4958
              "It's not great."
              ; => 2.7777
              "I didn't really like this."
              ; => 2.6690
              "Worth it! I recommend this to all my friends!"
              ; => 4.9123
              "I love this book."
              ; => 5.0000
              "I didn't like the main character, but I didn't feel the same way about the book."
              ; => 2.3010
              "This is a bad book."
              ; => 1.3488
              "When I'm having a bad day, this is the book I read."
              ; => 4.4020
              "Some people like this, but I am not one of them."
              ; => 3.3037
              ]
             dataset-config
             trained-model)
    ;; => A Tensure vector, one entry for each string (outputs for each string are next to the string above).
    ))
