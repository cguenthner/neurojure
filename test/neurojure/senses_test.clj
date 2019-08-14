(ns neurojure.senses-test
  (:require [clojure.test :refer :all]
            [neurojure.senses :refer :all]
            [neurojure.utils :as u]
            [tensure.core :as m]))

(defn valid-one-hot-code?
  [encoded]
  (->> (m/eseq encoded)
       (every? #(or (= 0.0 %) (= 1.0 %)))))

(deftest make-one-hot-code-test
  (testing "no other? category"
    (let [{:keys [encode decode]} (make-one-hot-code [:a :b :b])
          encoded1 (encode [:a])
          encoded4 (encode [:a :b :b :a])]
      (is (valid-one-hot-code? encoded4))
      (is (= (m/shape encoded4) [4 2]))
      (is (= (m/shape encoded1) [1 2]))
      (is (= (decode encoded1) [:a]))
      (is (= (decode encoded4) [:a :b :b :a]))
      (is (thrown? Exception (encode [:a :b :c])))))
  (testing "other? with default other-sentinel"
    (let [{:keys [encode decode]} (make-one-hot-code [:a :b :b] {:other? true})
          encoded1a (encode [:a])
          encoded1c (encode [:c])
          encoded4 (encode [:c :b :d :a])]
      (is (valid-one-hot-code? encoded4))
      (is (= (m/shape encoded1a) [1 3]))
      (is (= (m/shape encoded1c) [1 3]))
      (is (= (m/shape encoded4) [4 3]))
      (is (= (decode encoded1a) [:a]))
      (is (= (decode encoded1c) [-1]))
      (is (= (decode encoded4) [-1 :b -1 :a]))))
  (testing "other? with custom sentinel"
    (let [{:keys [encode decode]} (make-one-hot-code [:a :b] {:other? true
                                                              :other-sentinel :unk})]
      (is (valid-one-hot-code? (encode [:a :b :c :d])))
      (is (= (m/shape (encode [:a :b :c :d])) [4 3]))
      (is (= (decode (encode [:c :a :b :d])) [:unk :a :b :unk]))
      (is (= (decode (encode [:e :f])) [:unk :unk]))))
  (testing "works with a variety of hashable objects"
    (let [{:keys [encode decode]} (make-one-hot-code [:a "b" 2 3.14 3.14 {:q 4} #{:z} [6] '() {:q 4} #{[{8 '(9)}]}]
                                                     {:other? true})
          encoded (encode ["b" :a 2 "c" #{} #{[{8 '(9)}]} {:q 4} 3.14 #{:z} [6] 6.033 '() '(3)])]
      (is (valid-one-hot-code? encoded))
      (is (= (m/shape encoded) [13 10]))
      (is (= (decode encoded) ["b" :a 2 -1 -1 #{[{8 '(9)}]} {:q 4} 3.14 #{:z} [6] -1 '() -1])))))
