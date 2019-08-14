(ns neurojure.test-utils
  (:require [neurojure.core :refer [make-model evaluate-model]]
            [ranvier.core :refer [G]]
            [tensure.core :as m]))

(defn test-regularizer
  "A regularizer that adds the count of regularized parameter elements in the model. The output of a model
  using this regularizer can be compared to an unregulared model to confirm that the layers are applying the
  expected regularization."
  [g parameter-node]
  (#'neurojure.core/create-layer
   :test-regularizer
   [g parameter-node]
   (G (+ g (size parameter-node)))))

(defn get-regularization-diff
  "Returns the difference between the result of evaluating `regularized-graph` and `unregularized-graph` on
  `data`, a map of graph inputs to values."
  [unregularized-graph regularized-graph data]
  (let [regularized-model (make-model :data {:d data}
                                      :graph regularized-graph)
        unregularized-model (make-model :data {:d data}
                                        :graph unregularized-graph)]
    (m/sub (evaluate-model regularized-model :d {:training true})
           (evaluate-model unregularized-model :d {:training true}))))
