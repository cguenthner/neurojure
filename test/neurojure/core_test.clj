(ns neurojure.core-test
  (:require [clojure.test :refer [deftest is testing]]
            [neurojure.core :refer :all]
            [neurojure.test-utils :refer :all]
            [ranvier.core :as r :refer [G evaluate forward backward]]
            [ranvier.test-utils :refer :all]
            [tensure.core :as m]))

;; Function tests
(deftest mean-test
  (testing "vector - collapse"
    (is (nd= (evaluate (mean [1 2 3] {:axis 0 :collapse true})) 2)))
  (testing "vector - keep dims"
    (is (nd= (evaluate (mean [1 2 3] {:axis 0})) [2])))
  (testing "matrix - collapse"
    (is (nd= (evaluate (mean [[1 2 3] [4 5 6]] {:axis 0 :collapse true})) [2.5 3.5 4.5]))
    (is (nd= (evaluate (mean [[1 2 3] [4 5 6]] {:axis 1 :collapse true})) [2 5]))
    (is (about= (evaluate (mean [[1 2 3] [4 5 6]] {:axis [0 1] :collapse true})) 3.5))
    (is (about= (evaluate (mean [[1 2 3] [4 5 6]] {:collapse true})) 3.5)))
  (testing "matrix - keep dims"
    (is (nd= (evaluate (mean [[1 2 3] [4 5 6]] {:axis 0})) [[2.5 3.5 4.5]]))
    (is (nd= (evaluate (mean [[1 2 3] [4 5 6]] {:axis 1})) [[2] [5]]))
    (is (about= (evaluate (mean [[1 2 3] [4 5 6]] {:axis [0 1]})) [[3.5]]))
    (is (about= (evaluate (mean [[1 2 3] [4 5 6]] {:collapse false})) [[3.5]])))
  (testing "tensor - collapse"
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis 0
                                                     :collapse true})) [[4 5 6] [7 8 9]]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis 1
                                                     :collapse true})) [[2.5 3.5 4.5] [8.5 9.5 10.5]]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis 2
                                                     :collapse true})) [[2 5] [8 11]]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis [0 1]
                                                     :collapse true})) [5.5 6.5 7.5]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis [0 2]
                                                     :collapse true})) [5 8]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis [1 2]
                                                     :collapse true})) [3.5 9.5]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis [0 1 2]
                                                     :collapse true})) 6.5))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:collapse true})) 6.5)))
  (testing "tensor - keep dims"
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis 0})) [[[4 5 6] [7 8 9]]]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis 1})) [[[2.5 3.5 4.5]] [[8.5 9.5 10.5]]]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis 2})) [[[2] [5]] [[8] [11]]]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis [0 1]})) [[[5.5 6.5 7.5]]]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis [0 2]})) [[[5] [8]]]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis [1 2]})) [[[3.5]] [[9.5]]]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:axis [0 1 2]})) [[[6.5]]]))
    (is (nd= (evaluate (mean [[[1 2 3] [4 5 6]]
                              [[7 8 9] [10 11 12]]] {:collapse false})) [[[6.5]]]))))

(deftest normalize-test
  (is (nd= (evaluate (normalize [[-3 0 3] [-30 0 30]] {:axis 1}))
           [[-0.5 0 0.5] [-0.05 0 0.05]]))
  (is (nd= (evaluate (normalize [[-3 -30] [0 0] [3 30]] {:axis 0}))
           [[-0.5 -0.05] [0 0] [0.5 0.05]]))
  (is (nd= (evaluate (normalize [[0 3 6] [-6 -3 0] [-3 0 3]] {:axis 1}))
           [[-0.5 0 0.5] [-0.5 0 0.5] [-0.5 0 0.5]])))

;; Initializer tests
(deftest zeros-initializer-test
  (check-tensor-creator zeros-initializer)
  (is (->> (zeros-initializer [10 10])
           m/eseq
           (every? zero?))))

(deftest ones-initializer-test
  (check-tensor-creator ones-initializer)
  (is (->> (ones-initializer [10 10])
           m/eseq
           (every? #(nd= % 1)))))

(deftest make-fill-initializer-test
  (let [pi-initializer (make-fill-initializer 3.14)]
    (check-tensor-creator pi-initializer)
    (is (->> (pi-initializer [10 10])
             m/eseq
             (every? #(nd= % 3.14))))))

(deftest make-constant-initializer-test
  (testing "returns an initializer that returns the constant"
    (is (nd= ((make-constant-initializer 7) nil)
             7))
    (is (nd= ((make-constant-initializer [1 2 3]) [3])
             [1 2 3]))
    (is (nd= ((make-constant-initializer [[1 2 3] [4 5 6]]) [2 3])
             [[1 2 3] [4 5 6]]))
    (is (nd= ((make-constant-initializer [[[1 2 3] [4 5 6]]
                                          [[7 8 9] [10 11 12]]]) [2 2 3])
             [[[1 2 3] [4 5 6]]
              [[7 8 9] [10 11 12]]])))
  (testing "throws an exception if shapes do not match"
    (is (thrown? Exception ((make-constant-initializer 7) [1 1])))
    (is (thrown? Exception ((make-constant-initializer [[1 2 3] [4 5 6]]) [3 2])))))

(deftest make-uniform-random-initializer-test
  (r/set-rng-seed! 0)
  (let [fa (make-uniform-random-initializer -0.1 0.3)
        a (fa [10000])
        b ((make-uniform-random-initializer (constantly 5) (constantly 10)) [10000])]
    (check-tensor-creator fa)
    (testing "produces results with the correct statistical properities"
      (is (about= (m/emean a) 0.1 0.01))
      (is (about= (m/emean b) 7.5 0.1))
      (is (about= (m/estdev a) (uniform-sd -0.1 0.3) 0.001))
      (is (about= (m/estdev b) (uniform-sd 5 10) 0.01))
      (is (>= (m/scalar->number (m/emin a)) -0.1))
      (is (< (m/scalar->number (m/emax a))) 0.3)
      (is (>= (m/scalar->number (m/emin b)) 5))
      (is (< (m/scalar->number (m/emax b)) 10)))))

(deftest make-normal-random-initializer-test
  (r/set-rng-seed! 0)
  (let [fa (make-normal-random-initializer 0 0.01)
        a (fa [10000])
        b ((make-normal-random-initializer (constantly 5) (constantly 10)) [10000])]
    (check-tensor-creator fa)
    (testing "produces results with the correct statistical properities"
      (is (about= (m/emean a) 0 0.001))
      (is (about= (m/emean b) 5 0.2))
      (is (about= (m/estdev a) 0.01 0.001))
      (is (about= (m/estdev b) 10 0.1)))))

; The SD of a normal distribution with SD 1 truncated +/- 2 SD's from the mean will be ~0.88.
; See https://en.wikipedia.org/wiki/Truncated_normal_distribution.
; Z = 0.958, since +/- 2 SD's include 95.8% of samples.
; alpha = -2, beta = 2
; phi(alpha) = phi(beta) = (/ (Math/exp -2) (Math/sqrt (* 2 Math/PI))) = ~0.05399
; SD = (Math/sqrt (- 1 (/ (* 4 0.05399) 0.958))) = ~0.88
(deftest make-truncated-random-initializer-test
  (r/set-rng-seed! 0)
  (let [fa (make-truncated-random-initializer 0 0.01)
        a (fa [1e4])
        b ((make-truncated-random-initializer (constantly 5) (constantly 10)) [1e4])]
    (check-tensor-creator fa)
    (testing "produces results with the correct statistical properities"
      (is (about= (m/emean a) 0 0.01))
      (is (about= (m/emean b) 5 1.5))
      (is (about= (m/estdev a) 0.0088 0.001))
      (is (about= (m/estdev b) 8.88 0.3))
      (is (> (m/scalar->number (m/emin a)) -0.02))
      (is (< (m/scalar->number (m/emax a)) 0.02))
      (is (> (m/scalar->number (m/emin b)) -15))
      (is (< (m/scalar->number (m/emax b)) 25)))))

(deftest make-variance-scaling-initializer-test
  (r/set-rng-seed! 0)
  (let [var-1-axis-0 (make-variance-scaling-initializer 1 0)
        var-1-axis-1 (make-variance-scaling-initializer 1 1)
        v1a1-values (var-1-axis-1 [1 10000])
        v4a1-values ((make-variance-scaling-initializer 4 1) [1 10000])
        v1a01-values ((make-variance-scaling-initializer 1 [0 1]) [100 100])]
    (testing "returns tensors of the correct shape"
      (is (= (m/shape (var-1-axis-0 [3])) [3]))
      (is (= (m/shape (var-1-axis-1 [2 3])) [2 3]))
      (is (= (m/shape (var-1-axis-1 [2 3 4])) [2 3 4])))
    (testing "scales based on the correct axis"
      (is (about= (m/estdev (var-1-axis-0 [10000 1])) 0.0088 0.001))
      (is (about= (m/estdev (var-1-axis-0 [1 10000])) 0.88 0.1))
      (is (about= (m/estdev (var-1-axis-1 [10000 1])) 0.88 0.1))
      (is (about= (m/estdev v1a1-values) 0.0088 0.001))
      (is (about= (m/estdev v1a01-values) 0.071 0.01)))
    (testing "has the correct statistical properties"
      (is (about= (m/estdev v4a1-values) 0.0176 0.001))
      (is (about= (m/emean v1a1-values) 0 0.01))
      (is (about= (m/emean v4a1-values) 0 0.01))
      (is (about= (m/emean v1a01-values) 0 0.01))
      (is (> (m/scalar->number (m/emin v1a1-values)) -0.02))
      (is (< (m/scalar->number (m/emax v1a1-values)) 0.02))
      (is (> (m/scalar->number (m/emin v4a1-values)) -0.04))
      (is (< (m/scalar->number (m/emax v4a1-values)) 0.04))
      (is (> (m/scalar->number (m/emin v1a01-values)) -0.142))
      (is (< (m/scalar->number (m/emax v1a01-values)) 0.142)))))

(deftest make-lecun-uniform-initializer-test
  (r/set-rng-seed! 0)
  (let [f (make-lecun-uniform-initializer)
        nd (f [1 10800])]
    (testing "returns tensors of the correct shape"
      (is (= (m/shape (f [2 3])) [2 3]))
      (is (= (m/shape (f [2 3 4])) [2 3 4])))
    (testing "returns tensors with the expected statistical properties"
      (is (>= (m/scalar->number (m/emin nd)) -1/60))
      (is (< (m/scalar->number (m/emax nd)) 1/60))
      (is (about= (m/emean nd) 0 0.001))
      (is (about= (m/estdev nd) (uniform-sd -1/60 1/60) 0.0001)))))

(deftest make-lecun-normal-initializer-test
  (r/set-rng-seed! 0)
  (let [f (make-lecun-normal-initializer)
        nd (f [1 10000])]
    (testing "returns tensors of the correct shape"
      (is (= (m/shape (f [2 3])) [2 3]))
      (is (= (m/shape (f [2 3 4])) [2 3 4])))
    (testing "returns tensors with the expected statistical properties"
      (is (> (m/scalar->number (m/emin nd)) -0.02))
      (is (< (m/scalar->number (m/emax nd)) 0.02))
      (is (about= (m/emean nd) 0 0.001))
      (is (about= (m/estdev nd) 0.0088 0.0001)))))

(deftest make-he-uniform-initializer-test
  (r/set-rng-seed! 0)
  (let [f (make-he-uniform-initializer)
        nd (f [1 9600])]
    (testing "returns tensors of the correct shape"
      (is (= (m/shape (f [2 3])) [2 3]))
      (is (= (m/shape (f [2 3 4])) [2 3 4])))
    (testing "returns tensors with the expected statistical properties"
      (is (>= (m/scalar->number (m/emin nd)) -1/40))
      (is (< (m/scalar->number (m/emax nd)) 1/40))
      (is (about= (m/emean nd) 0 0.001))
      (is (about= (m/estdev nd) (uniform-sd -1/40 1/40) 0.0001)))))

(deftest make-he-normal-initializer-test
  (r/set-rng-seed! 0)
  (let [f (make-he-normal-initializer)
        nd (f [1 9800])]
    (testing "returns tensors of the correct shape"
      (is (= (m/shape (f [2 3])) [2 3]))
      (is (= (m/shape (f [2 3 4])) [2 3 4])))
    (testing "returns tensors with the expected statistical properties"
      (is (> (m/scalar->number (m/emin nd)) -1/35))
      (is (< (m/scalar->number (m/emax nd)) 1/35))
      (is (about= (m/emean nd) 0 0.001))
      (is (about= (m/estdev nd) (* 0.88 1/70) 0.0001)))))

(deftest make-xavier-uniform-initializer-test
  (r/set-rng-seed! 0)
  (let [f (make-xavier-uniform-initializer)
        nd (f [108 108])]
    (testing "returns tensors of the correct shape"
      (is (= (m/shape (f [2 3])) [2 3]))
      (is (= (m/shape (f [2 3 4])) [2 3 4])))
    (testing "returns tensors with the expected statistical properties"
      (is (>= (m/scalar->number (m/emin nd)) -1/6))
      (is (< (m/scalar->number (m/emax nd)) 1/6))
      (is (about= (m/emean nd) 0 0.01))
      (is (about= (m/estdev nd) (uniform-sd -1/6 1/6) 0.01)))))

(deftest make-xavier-normal-initializer-test
  (r/set-rng-seed! 0)
  (let [f (make-xavier-normal-initializer)
        nd (f [100 100])]
    (testing "returns tensors of the correct shape"
      (is (= (m/shape (f [2 3])) [2 3]))
      (is (= (m/shape (f [2 3 4])) [2 3 4])))
    (testing "returns tensors with the expected statistical properties"
      (is (> (m/scalar->number (m/emin nd)) -0.2))
      (is (< (m/scalar->number (m/emax nd)) 0.2))
      (is (about= (m/emean nd) 0 0.01))
      (is (about= (m/estdev nd) 0.088 0.01)))))

;; Regularizer tests
; The `make-l2-regularization-test` serve both a unit tests for that particular regularizer as well as an
; integration tests of the overall regularization system.
(deftest make-l2-regularizer-test
  (testing "adds the correct regularization term for a single parameter"
    (let [get-diff (fn [value num-examples regularization-weight]
                     (get-regularization-diff
                       (G (esum (#'neurojure.core/make-parameter-node :test-param nil nil)))
                       (G (regularize
                            (esum (#'neurojure.core/make-parameter-node
                                   :test-param nil nil
                                   {:regularizer (make-l2-regularizer num-examples regularization-weight)}))))
                       {:test-param value}))]
      (is (nd= (get-diff 7 1 2) 49))
      (is (nd= (get-diff [1 2 3] 3 6) 14))
      (is (nd= (get-diff [[1 2] [3 4]] 2 4) 30))
      (is (nd= (get-diff [[[1 2] [3 4]] [[5 6] [7 8]]] 2 4) 204))
      (is (nd= (get-diff [[1 2] [3 4]] 4 4) 15))
      (is (nd= (get-diff [[1 2] [3 4]] 2 8) 60))))
  (testing "counts each parameter only once"
    (let [unregularized-param (#'neurojure.core/make-parameter-node :test-param nil nil)
          regularized-param (#'neurojure.core/make-parameter-node :test-param nil nil
                             {:regularizer (make-l2-regularizer 2 4)})
          regularized-graph (G (regularize
                                 (esum (* (+ regularized-param regularized-param) regularized-param))))
          unregularized-graph (G (esum (* (+ unregularized-param unregularized-param) unregularized-param)))]
      (is (nd= (get-regularization-diff unregularized-graph regularized-graph
                                        {:test-param [[1 2] [3 4]]})
               30))))
  (testing "applies regularization to multiple parameters"
    (let [unregularized-param1 (#'neurojure.core/make-parameter-node :test-param1 nil nil)
          unregularized-param2 (#'neurojure.core/make-parameter-node :test-param2 nil nil)
          regularized-param1 (#'neurojure.core/make-parameter-node :test-param1 nil nil
                              {:regularizer (make-l2-regularizer 2 4)})
          regularized-param2 (#'neurojure.core/make-parameter-node :test-param2 nil nil
                              {:regularizer (make-l2-regularizer 2 4)})
          regularized-graph (G (regularize (esum (* (+ regularized-param1 7) regularized-param2))))
          unregularized-graph (G (esum (* (+ unregularized-param1 7) unregularized-param2)))]
      (is (nd= (get-regularization-diff unregularized-graph regularized-graph
                                        {:test-param1 [[1 2] [3 4]]
                                         :test-param2 [[4 5] [6 7]]})
               156)))))

(deftest make-l1-regularizer-test
  (testing "adds the correct regularization term for a single parameter"
    (let [get-diff (fn [value num-examples regularization-weight]
                     (get-regularization-diff
                       (G (esum (#'neurojure.core/make-parameter-node :test-param nil nil)))
                       (G (regularize
                            (esum (#'neurojure.core/make-parameter-node
                                   :test-param nil nil
                                   {:regularizer (make-l1-regularizer num-examples regularization-weight)}))))
                       {:test-param value}))]
      (is (nd= (get-diff 7 1 2) 7))
      (is (nd= (get-diff [1 2 3] 3 6) 6))
      (is (nd= (get-diff [[1 2] [3 4]] 2 4) 10))
      (is (nd= (get-diff [[[1 2] [3 4]] [[5 6] [7 8]]] 2 4) 36))
      (is (nd= (get-diff [[1 2] [3 4]] 4 4) 5))
      (is (nd= (get-diff [[1 2] [3 4]] 2 8) 20)))))

;; Layer tests
(deftest dense-test
  (testing "applies regularization"
    (let [regularized-graph (G (-> (dense :in {:units 1
                                               :initializer (make-fill-initializer 1)
                                               :regularizer test-regularizer})
                                   esum
                                   regularize))
          unregularized-graph (G (esum (dense :in {:units 1
                                                   :initializer (make-fill-initializer 1)})))]
      (is (nd= (get-regularization-diff unregularized-graph regularized-graph {:in [[1 2 3] [4 5 6]]})
               3))))
  (testing "initializes parameters correctly"
    (evaluates-to? (dense [[1 2 3] [4 5 6]] {:units 1 :initializer (make-fill-initializer 1)})
                   [[7] [16]])
    (evaluates-to? (dense [[1 2 3] [4 5 6]] {:units 1 :initializer (make-fill-initializer 2)})
                   [[14] [32]])
    (evaluates-to? (dense [[1 2 3] [4 5 6]] {:units 1
                                             :W-initializer (make-fill-initializer 1)
                                             :b-initializer (make-constant-initializer [[3]])})
                   [[9] [18]]))
  (testing "produces the expected result"
    (let [g (dense :in {:units 1
                        :initializer (make-fill-initializer 1)})
          inputs {:in [[1 2 3] [4 5 6]]}]
      (evaluates-to? g inputs [[7] [16]])
      (params-numerically-validated? g inputs))
    (let [g (dense :in {:units 3
                        :W-initializer (make-constant-initializer [[0 3 1]
                                                                   [1 2 1]
                                                                   [2 -1 1]])
                        :b-initializer (make-constant-initializer [[1 2 3]])})
          inputs {:in [[1 2 3] [4 5 6]]}]
      (evaluates-to? g inputs [[9 6 9] [18 18 18]])
      (params-numerically-validated? g inputs))))

(deftest dropout-test
  (r/set-rng-seed! 0)
  (testing "scaling"
    (evaluates-to-about? (G (esum (dropout (ones [1000]) {:frequency 0.5})))
                         {:training 1} 1000)
    (evaluates-to-about? (G (esum (dropout (ones [1000]) {:frequency 0.2})))
                         {:training 1} 1000)
    (is (->> (evaluate (G (dropout (ones [100]) {:frequency 0.5 :exact false}))
                       {:training 1})
             m/eseq
             (every? #(or (nd= 0 %) (nd= 2 %)))))
    (is (->> (evaluate (G (dropout (ones [100]) {:frequency 0.2 :exact false}))
                       {:training 1})
             m/eseq
             (every? #(or (nd= 0 %) (nd= 1.25 %))))))
  (testing "dropout during training but not testing"
    (let [g (G (dropout (ones [100]) {:frequency 0.5}))]
      (is (not (nd= (evaluate g {:training 1})
                    (m/ones [100]))))
      (is (nd= (evaluate g {:training 0})
               (m/ones [100])))))
  (testing "different shaped tensors"
    ; Exact dropout scaling on scalars is numerically unstable.
    #_(let [freqs (->> (repeatedly 1000 #(evaluate (G (dropout 3 {:frequency 0.5 :exact false}))
                                                   {:training 1}))
                       frequencies)]
        ; These assertions are non-deterministic. They could in theory fail on occasion even though everything
        ; is working fine.
        (is (between? 200 (get freqs 0.0) 700))
        (is (between? 200 (get freqs 6.0) 700)))
    (let [result (evaluate (G (dropout (ones [25 25]) {:frequency 0.5}))
                           {:training 1})]
      (is (= (m/shape result) [25 25]))
      (is (between? 100 (->> result m/eseq (filter #(nd= % 0)) count) 400)))
    (let [result (evaluate (G (dropout (ones [10 10 10]) {:frequency 0.5}))
                           {:training 1})]
      (is (= (m/shape result) [10 10 10]))
      (is (between? 200 (->> result m/eseq (filter #(nd= % 0)) count) 800))))
  (testing "backpropagation"
    (let [g (G (* 7 (dropout :a {:frequency 0.5})))
          input-map {:training 1 :a (m/filled [100] 2)}
          values (forward g input-map)
          result (get values (r/get-node-name g))
          grad (:a (backward g values [:a]))]
      (is (->> (m/emap
                 #(if (or (and (about= %1 0)
                               (about= %2 0))
                          (about= (/ %1 %2) 2))
                    1
                    0)
                 result grad)
               (about= (m/ones [100])))))))

(deftest conv2d-test
  (let [in [[[[1 1 2] [3 2 1] [1 6 2] [7 1 4]]
             [[4 0 1] [3 2 1] [1 2 3] [7 3 1]]
             [[2 0 1] [0 0 0] [2 2 6] [3 3 3]]
             [[1 1 1] [0 1 0] [2 4 2] [4 4 2]]]
            [[[3 0 4] [7 6 4] [0 0 0] [1 1 2]]
             [[3 3 0] [6 4 3] [8 3 1] [0 1 3]]
             [[3 3 1] [2 2 2] [3 0 1] [4 5 2]]
             [[1 2 1] [1 1 1] [0 0 1] [6 3 2]]]]]
    ; This kernel sums the first column of the first three rows.
    (let [[out [din dkernel]] (test-op conv2d [[[[1]]]
                                               [[[3]]]]
                                       in
                                       [[[[1 1 1]]
                                         [[1 1 1]]
                                         [[1 1 1]]]]
                                       [2 5])]
      (is (nd= out [[[[12]]] [[[20]]]]))
      (is (nd= din [[[[1 1 1] [0 0 0] [0 0 0] [0 0 0]]
                     [[1 1 1] [0 0 0] [0 0 0] [0 0 0]]
                     [[1 1 1] [0 0 0] [0 0 0] [0 0 0]]
                     [[0 0 0] [0 0 0] [0 0 0] [0 0 0]]]
                    [[[3 3 3] [0 0 0] [0 0 0] [0 0 0]]
                     [[3 3 3] [0 0 0] [0 0 0] [0 0 0]]
                     [[3 3 3] [0 0 0] [0 0 0] [0 0 0]]
                     [[0 0 0] [0 0 0] [0 0 0] [0 0 0]]]]))
      (is (nd= dkernel [[[[10 1 14]]
                         [[13 9 1]]
                         [[11 9 4]]]])))
    ; This kernel sum across channels and columns within a single row.
    (let [[out [din dkernel]] (test-op conv2d
                                       [[[[1]] [[0]]]
                                        [[[3]] [[2]]]]
                                       in
                                       [[[[1 1 1] [1 1 1] [1 1 1] [1 1 1]]]]
                                       [2 50])]
      (is (nd= out [[[[31]] [[22]]]
                    [[[28]] [[28]]]]))
      (is (nd= din [[[[1 1 1] [1 1 1] [1 1 1] [1 1 1]]
                     [[0 0 0] [0 0 0] [0 0 0] [0 0 0]]
                     [[0 0 0] [0 0 0] [0 0 0] [0 0 0]]
                     [[0 0 0] [0 0 0] [0 0 0] [0 0 0]]]
                    [[[3 3 3] [3 3 3] [3 3 3] [3 3 3]]
                     [[0 0 0] [0 0 0] [0 0 0] [0 0 0]]
                     [[2 2 2] [2 2 2] [2 2 2] [2 2 2]]
                     [[0 0 0] [0 0 0] [0 0 0] [0 0 0]]]]))
      (is (nd= dkernel [[[[16 7 16] [28 24 17] [7 6 4] [18 14 14]]]])))
    ; This kernel will select the third channel and the first channel.
    (let [[out [din dkernel]] (test-op conv2d [[[[1 0] [2 4] [3 3] [4 1]]
                                                [[1 1] [3 5] [2 4] [5 1]]
                                                [[2 2] [2 3] [3 4] [0 0]]
                                                [[0 1] [1 1] [4 2] [3 2]]]
                                               [[[3 4] [4 4] [1 1] [3 2]]
                                                [[4 2] [1 4] [1 4] [1 3]]
                                                [[0 3] [0 1] [0 2] [1 4]]
                                                [[4 1] [4 1] [4 1] [3 2]]]]
                                       in
                                       [[[[0 0 1]]] [[[1 0 0]]]] [1 1])]
      (is (nd= out [[[[2 1] [1 3] [2 1] [4 7]]
                     [[1 4] [1 3] [3 1] [1 7]]
                     [[1 2] [0 0] [6 2] [3 3]]
                     [[1 1] [0 0] [2 2] [2 4]]]
                    [[[4 3] [4 7] [0 0] [2 1]]
                     [[0 3] [3 6] [1 8] [3 0]]
                     [[1 3] [2 2] [1 3] [2 4]]
                     [[1 1] [1 1] [1 0] [2 6]]]]))
      (is (nd= din [[[[0 0 1] [4 0 2] [3 0 3] [1 0 4]]
                     [[1 0 1] [5 0 3] [4 0 2] [1 0 5]]
                     [[2 0 2] [3 0 2] [4 0 3] [0 0 0]]
                     [[1 0 0] [1 0 1] [2 0 4] [2 0 3]]]
                    [[[4 0 3] [4 0 4] [1 0 1] [2 0 3]]
                     [[2 0 4] [4 0 1] [4 0 1] [3 0 1]]
                     [[3 0 0] [1 0 0] [2 0 0] [4 0 1]]
                     [[1 0 4] [1 0 4] [1 0 4] [2 0 3]]]]))
      (is (nd= dkernel [[[[214 160 136]]] [[[228 177 151]]]])))
    ; 2x2 version of the kernel immediately above.
    (let [[out [din dkernel]] (test-op conv2d [[[[1 0] [2 4] [3 3]]
                                                [[1 1] [3 5] [2 4]]
                                                [[2 2] [2 3] [3 4]]]
                                               [[[3 4] [4 4] [1 1]]
                                                [[4 2] [1 4] [1 4]]
                                                [[0 3] [0 1] [0 2]]]]
                                       in
                                       [[[[0 0 1] [0 0 1]]
                                         [[0 0 1] [0 0 1]]]
                                        [[[1 0 0] [1 0 0]]
                                         [[1 0 0] [1 0 0]]]] [1 1])]
      (is (nd= out [[[[5 11] [7 8] [10 16]]
                     [[3 9] [10 6] [13 13]]
                     [[2 3] [8 4] [13 11]]]
                    [[[11 19] [8 21] [6 9]]
                     [[6 14] [7 19] [7 15]]
                     [[5 7] [5 6] [6 13]]]]))
      (is (nd= din [[[[0 0 1] [4 0 3] [7 0 5] [3 0 3]]
                     [[1 0 2] [10 0 7] [16 0 10] [7 0 5]]
                     [[3 0 3] [11 0 8] [16 0 10] [8 0 5]]
                     [[2 0 2] [5 0 4] [7 0 5] [4 0 3]]]
                    [[[4 0 3] [8 0 7] [5 0 5] [1 0 1]]
                     [[6 0 7] [14 0 12] [13 0 7] [5 0 2]]
                     [[5 0 4] [10 0 5] [11 0 2] [6 0 1]]
                     [[3 0 0] [4 0 0] [3 0 0] [2 0 0]]]]))
      (is (nd= dkernel [[[[98 82 72] [113 83 80]]
                         [[85 72 53] [119 87 71]]]
                        [[[169 121 109] [169 134 121]]
                         [[109 95 81] [172 131 114]]]])))
    ; Same as immediately above but with stride [2 2] instead of stride [1 1].
    (let [[out [din dkernel]] (test-op conv2d [[[[1 0] [3 3]]
                                                [[2 2] [3 4]]]
                                               [[[3 4] [1 1]]
                                                [[0 3] [0 2]]]]
                                       in
                                       [[[[0 0 1] [0 0 1]]
                                         [[0 0 1] [0 0 1]]]
                                        [[[1 0 0] [1 0 0]]
                                         [[1 0 0] [1 0 0]]]] [2 2])]
      (is (nd= out [[[[5 11] [10 16]]
                     [[2 3] [13 11]]]
                    [[[11 19] [6 9]]
                     [[5 7] [6 13]]]]))
      (is (nd= din [[[[0 0 1] [0 0 1] [3 0 3] [3 0 3]]
                     [[0 0 1] [0 0 1] [3 0 3] [3 0 3]]
                     [[2 0 2] [2 0 2] [4 0 3] [4 0 3]]
                     [[2 0 2] [2 0 2] [4 0 3] [4 0 3]]]
                    [[[4 0 3] [4 0 3] [1 0 1] [1 0 1]]
                     [[4 0 3] [4 0 3] [1 0 1] [1 0 1]]
                     [[3 0 0] [3 0 0] [2 0 0] [2 0 0]]
                     [[3 0 0] [3 0 0] [2 0 0] [2 0 0]]]]))
      (is (nd= dkernel [[[[23 25 40] [55 33 36]]
                         [[32 32 19] [54 38 22]]]
                        [[[42 35 53] [76 56 52]]
                         [[36 45 25] [76 53 33]]]])))))

; TODO: Fix tolerances. Tolerances below are very for some values, because different tolerances can't be specified
; for different values in maps for the numerical validation routines. Maybe specify tolerances as a fraction with
; a min absolute tolerance difference.
(deftest conv2d-layer-test
  (r/set-rng-seed! 1)
  (let [a [[[[1 1 2] [3 2 1] [1 6 2] [7 1 4]]
            [[4 0 1] [3 2 1] [1 2 3] [7 3 1]]
            [[2 0 1] [0 0 0] [2 2 6] [3 3 3]]
            [[1 1 1] [0 1 0] [2 4 2] [4 4 2]]]
           [[[3 0 4] [7 6 4] [0 0 0] [1 1 2]]
            [[3 3 0] [6 4 3] [8 3 1] [0 1 3]]
            [[3 3 1] [2 2 2] [3 0 1] [4 5 2]]
            [[1 2 1] [1 1 1] [0 0 1] [6 3 2]]]]
        b [[[[1 1 2] [3 2 1]]
            [[4 0 1] [3 2 1]]]
           [[[3 0 4] [7 6 4]]
            [[3 3 0] [6 4 3]]]
           [[[1 6 2] [7 1 4]]
            [[1 2 3] [7 3 1]]]
           [[[2 2 6] [3 3 3]]
            [[2 4 2] [4 4 2]]]]
        has-expected-shape? (fn [g in shape]
                              (is (= (m/shape (evaluate g {:in in})) shape)))]
    (testing "applies regularization"
      (let [options {:output-channels 2
                     :kernel-size [2 2]
                     :strides [2 2]
                     :initializer (make-fill-initializer 1)}
            regularized-graph (G (regularize (esum (conv2d-layer :in (assoc options :regularizer test-regularizer)))))
            unregularized-graph (G (esum (conv2d-layer :in options)))]
        (is (nd= (get-regularization-diff unregularized-graph regularized-graph {:in a})
                 24))
        (is (nd= (get-regularization-diff unregularized-graph regularized-graph {:in b})
                 24))))
    (testing "throws and exception when kernel-size is invalid"
      (is (thrown? Exception (evaluate (conv2d-layer a {:kernel-size [1]
                                                        :output-channels 2})))))
    (testing "throws and exception when strides is invalid"
      (is (thrown? Exception (evaluate (conv2d-layer a {:kernel-size [1 1]
                                                        :output-channels 2
                                                        :strides 2})))))
    (testing "throws and exception when padding is invalid"
      (is (thrown? Exception (evaluate (conv2d-layer a {:kernel-size [1 1]
                                                        :output-channels 2
                                                        :padding [1]})))))
    (testing "works for different kernel sizes and output channels"
      (let [g (conv2d-layer :in {:kernel-size [1 1]
                                 :output-channels 2
                                 :kernel-initializer (make-fill-initializer 1)
                                 :bias-initializer (make-fill-initializer 2)})]
        (evaluates-to? g {:in b} [[[[6 6] [8 8]] [[7 7] [8 8]]]
                                  [[[9 9] [19 19]] [[8 8] [15 15]]]
                                  [[[11 11] [14 14]] [[8 8] [13 13]]]
                                  [[[12 12] [11 11]] [[10 10] [12 12]]]])
        (params-numerically-validated? g {:in b} 0.5))
      (let [g (conv2d-layer :in {:kernel-size [2 2]
                                 :output-channels 2
                                 :strides [2 2]
                                 :kernel-initializer (make-constant-initializer [[[[0 0 1] [0 0 1]]
                                                                                  [[0 0 1] [0 0 1]]]
                                                                                 [[[1 0 0] [1 0 0]]
                                                                                  [[1 0 0] [1 0 0]]]])
                                 :bias-initializer (make-constant-initializer [[[[1 2]]]])})]
        (evaluates-to? g {:in a} [[[[6 13] [11 18]]
                                   [[3 5] [14 13]]]
                                  [[[12 21] [7 11]]
                                   [[6 9] [7 15]]]])
        (params-numerically-validated? g {:in b} 0.5))
      (let [g (conv2d-layer :in {:kernel-size [2 1]
                                 :output-channels 5
                                 :initializer (make-xavier-normal-initializer)})]
        (has-expected-shape? g b [4 1 2 5])
        (params-numerically-validated? g {:in b} 0.5))
      (let [g (conv2d-layer :in {:kernel-size [1 2]
                                 :output-channels 1
                                 :initializer (make-xavier-normal-initializer)})]
        (has-expected-shape? g b [4 2 1 1])
        (params-numerically-validated? g {:in b} 0.5))
      (let [g (conv2d-layer :in {:kernel-size [2 2]
                                 :output-channels 3
                                 :initializer (make-xavier-normal-initializer)})]
        (has-expected-shape? g b [4 1 1 3])
        (params-numerically-validated? g {:in b} 0.5)))
    (testing "works for different strides"
      (let [g (conv2d-layer :in {:kernel-size [1 1]
                                 :strides [2 2]
                                 :output-channels 1
                                 :initializer (make-xavier-normal-initializer)})]
        (has-expected-shape? g a [2 2 2 1])
        (params-numerically-validated? g {:in a} 0.5))
      (let [g (conv2d-layer :in {:kernel-size [2 2]
                                 :strides [2 2]
                                 :output-channels 1
                                 :initializer (make-xavier-normal-initializer)})]
        (has-expected-shape? g a [2 2 2 1])
        (params-numerically-validated? g {:in a} 0.5))
      (let [g (conv2d-layer :in {:kernel-size [1 1]
                                 :strides [2 1]
                                 :output-channels 3
                                 :initializer (make-xavier-normal-initializer)})]
        (has-expected-shape? g a [2 2 4 3])
        (params-numerically-validated? g {:in a} 0.5))
      (let [g (conv2d-layer :in {:kernel-size [1 2]
                                 :strides [4 1]
                                 :output-channels 2
                                 :initializer (make-xavier-normal-initializer)})]
        (has-expected-shape? g a [2 1 3 2])
        (params-numerically-validated? g {:in a} 0.5))
      (let [g (conv2d-layer :in {:kernel-size [3 2]
                                 :strides [2 3]
                                 :output-channels 1
                                 :initializer (make-xavier-normal-initializer)})]
        (has-expected-shape? g a [2 1 1 1])
        (params-numerically-validated? g {:in a} 0.5)))
    (testing "works with different paddings"
      (doseq [padding [[1 1] :same [[1 1] [1 1]]]]
        (let [g (conv2d-layer :in {:kernel-size [3 3]
                                   :padding padding
                                   :output-channels 3
                                   :initializer (make-xavier-normal-initializer)})]
          (has-expected-shape? g b [4 2 2 3])
          (params-numerically-validated? g {:in b} 0.5)))
      (doseq [padding [:same [[1 0] [1 0]]]]
        (let [g (conv2d-layer :in {:kernel-size [2 2]
                                   :padding padding
                                   :output-channels 1
                                   :initializer (make-xavier-normal-initializer)})]
          (has-expected-shape? g b [4 2 2 1])
          (params-numerically-validated? g {:in b} 0.5)))
      (let [g (conv2d-layer :in {:kernel-size [2 2]
                                 :padding [0 0]
                                 :output-channels 3
                                 :initializer (make-xavier-normal-initializer)})]
        (has-expected-shape? g b [4 1 1 3])
        (params-numerically-validated? g {:in b} 0.5))
      (doseq [padding [:same [[1 1] [1 0]]]]
        (let [g (conv2d-layer :in {:kernel-size [3 2]
                                   :strides [2 3]
                                   :padding padding
                                   :output-channels 1
                                   :initializer (make-xavier-normal-initializer)})]
          (has-expected-shape? g a [2 2 2 1])
          (params-numerically-validated? g {:in a} 0.5)))
      (let [g (conv2d-layer :in {:kernel-size [2 4]
                                 :strides [2 3]
                                 :padding [5 5]
                                 :output-channels 1
                                 :initializer (make-xavier-normal-initializer)})]
        (has-expected-shape? g b [4 6 3 1])
        (params-numerically-validated? g {:in a} 0.5)))
    (testing "initializes kernel correctly"
      (evaluates-to? (conv2d-layer [[[[1]]]] {:kernel-size [1 1]
                                              :output-channels 1
                                              :kernel-initializer (make-fill-initializer 7)
                                              :bias-initializer (make-fill-initializer 1)})
                     [[[[8]]]])
      (evaluates-to? (conv2d-layer [[[[1]]]] {:kernel-size [1 1]
                                              :output-channels 1
                                              :initializer (make-fill-initializer 3)})
                     [[[[6]]]]))))

(deftest max-pooling2d-test
  (let [a [[[[1 1 2] [3 2 1]]
            [[4 0 1] [3 2 1]]]
           [[[3 0 4] [7 6 4]]
            [[3 3 0] [6 4 3]]]]]
    (testing "throws an exception when pool-size is invalid"
      (is (thrown? Exception (evaluate (max-pooling2d a {:pool-size [1]})))))
    (testing "throws an exception when strides is invalid"
      (is (thrown? Exception (evaluate (max-pooling2d a {:pool-size [2 2]
                                                         :strides [1]})))))
    (testing "throws an exception when padding is invalid"
      (is (thrown? Exception (evaluate (max-pooling2d a {:pool-size [2 2]
                                                         :strides [1]})))))
    (testing "pools correctly with different pool sizes"
      (let [[v din] (test-op-with-args max-pooling2d [[[[1 2 3] [4 5 6]]
                                                       [[7 8 9] [10 11 12]]]
                                                      [[[13 14 15] [16 17 18]]
                                                       [[19 20 21] [22 23 24]]]]
                                       a
                                       {:pool-size [1 1]})]
        (is (nd= v a))
        (is (nd= din [[[[1 2 3] [4 5 6]]
                       [[7 8 9] [10 11 12]]]
                      [[[13 14 15] [16 17 18]]
                       [[19 20 21] [22 23 24]]]])))
      (let [[v din] (test-op-with-args max-pooling2d [[[[10 11 12]]]
                                                      [[[13 14 15]]]]
                                       a
                                       {:pool-size [2 2]})]
        (is (nd= v [[[[4 2 2]]]
                    [[[7 6 4]]]]))
        (is (nd= din [[[[0 0 12] [0 11 0]]
                       [[10 0 0] [0 11 0]]]
                      [[[0 0 15] [13 14 15]]
                       [[0 0 0] [0 0 0]]]])))
      (let [[v din] (test-op-with-args max-pooling2d [[[[10 11 12] [13 14 15]]]
                                                      [[[16 17 18] [19 20 21]]]]
                                       a
                                       {:pool-size [2 1]})]
        (is (nd= v [[[[4 1 2] [3 2 1]]]
                    [[[3 3 4] [7 6 4]]]]))
        (is (nd= din [[[[0 11 12] [13 14 15]]
                       [[10 0 0] [13 14 15]]]
                      [[[16 0 18] [19 20 21]]
                       [[16 17 0] [0 0 0]]]])))
      (let [[v din] (test-op-with-args max-pooling2d [[[[10 11 12]] [[13 14 15]]]
                                                      [[[16 17 18]] [[19 20 21]]]]
                                       a
                                       {:pool-size [1 2]})]
        (is (nd= v [[[[3 2 2]] [[4 2 1]]]
                    [[[7 6 4]] [[6 4 3]]]]))
        (is (nd= din [[[[0 0 12] [10 11 0]]
                       [[13 0 15] [0 14 15]]]
                      [[[0 0 18] [16 17 18]]
                       [[0 0 0] [19 20 21]]]]))))
    (testing "pool correctly with different padding"
      (let [[v din] (test-op-with-args max-pooling2d [[[[10 11 12]]]
                                                      [[[13 14 15]]]]
                                       a
                                       {:pool-size [2 2]
                                        :padding :same})]
        (is (nd= v [[[[1 1 2]]] [[[3 0 4]]]]))
        (is (nd= din [[[[10 11 12] [0 0 0]]
                       [[0 0 0] [0 0 0]]]
                      [[[13 14 15] [0 0 0]]
                       [[0 0 0] [0 0 0]]]])))
      (let [[v din] (test-op-with-args max-pooling2d [[[[1 2 3] [4 5 6]]
                                                       [[7 8 9] [10 11 12]]]
                                                      [[[13 14 15] [16 17 18]]
                                                       [[19 20 21] [22 23 24]]]]
                                       a
                                       {:pool-size [2 2]
                                        :padding [1 1]})]
        (is (nd= v [[[[1 1 2] [3 2 1]]
                     [[4 0 1] [3 2 1]]]
                    [[[3 0 4] [7 6 4]]
                     [[3 3 0] [6 4 3]]]]))
        (is (nd= din [[[[1 2 3] [4 5 6]]
                       [[7 8 9] [10 11 12]]]
                      [[[13 14 15] [16 17 18]]
                       [[19 20 21] [22 23 24]]]])))
      (let [[v din] (test-op-with-args max-pooling2d [[[[1 2 3] [4 5 6] [7 8 9]]
                                                       [[10 11 12] [13 14 15] [16 17 18]]
                                                       [[19 20 21] [22 23 24] [25 26 27]]]
                                                      [[[28 29 30] [31 32 33] [34 35 36]]
                                                       [[37 38 39] [40 41 42] [43 44 45]]
                                                       [[46 47 48] [49 50 51] [52 53 54]]]]
                                       a
                                       {:pool-size [2 2]
                                        :padding [[1 3] [1 3]]})]
        (is (nd= v [[[[1 1 2] [3 2 1] [0 0 0]]
                     [[4 0 1] [3 2 1] [0 0 0]]
                     [[0 0 0] [0 0 0] [0 0 0]]]
                    [[[3 0 4] [7 6 4] [0 0 0]]
                     [[3 3 0] [6 4 3] [0 0 0]]
                     [[0 0 0] [0 0 0] [0 0 0]]]]))
        (is (nd= din [[[[1 2 3] [4 5 6]]
                       [[10 11 12] [13 14 15]]]
                      [[[28 29 30] [31 32 33]]
                       [[37 38 39] [40 41 42]]]]))))
    (testing "pools correctly with different strides"
      (let [[v din] (test-op-with-args max-pooling2d [[[[1 2 3] [4 5 6] [7 8 9]]
                                                       [[10 11 12] [13 14 15] [16 17 18]]
                                                       [[19 20 21] [22 23 24] [25 26 27]]]
                                                      [[[28 29 30] [31 32 33] [34 35 36]]
                                                       [[37 38 39] [40 41 42] [43 44 45]]
                                                       [[46 47 48] [49 50 51] [52 53 54]]]]
                                       a
                                       {:pool-size [2 2]
                                        :strides [1 1]
                                        :padding [1 1]})]
        (is (nd= v [[[[1 1 2] [3 2 2] [3 2 1]]
                     [[4 1 2] [4 2 2] [3 2 1]]
                     [[4 0 1] [4 2 1] [3 2 1]]]
                    [[[3 0 4] [7 6 4] [7 6 4]]
                     [[3 3 4] [7 6 4] [7 6 4]]
                     [[3 3 0] [6 4 3] [6 4 3]]]]))
        (is (nd= din [[[[1 13 36] [27 44 27]]
                       [[64 20 45] [41 80 69]]]
                      [[[65 29 144] [148 152 156]]
                       [[83 85 48] [101 103 105]]]]))))))
