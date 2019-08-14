(defproject neurojure "0.0.1"
  :description "Clojure library for building neural network models"
  :url "https://github.com/cguenthner/neurojure"
  :license {:name "MIT License"
            :url "https://opensource.org/licenses/MIT"}
  :dependencies [[org.clojure/clojure "1.9.0"]
                 [tensure "0.1.0"]
                 [ranvier "0.1.1"]
                 [org.clojure/core.async "0.4.490"]
                 [org.clojure/data.json "0.2.6"]]
  :jvm-opts [; Increase heap size.
             "-Xmx16G"]
  :repl-options {:init-ns neurojure.core}
  :codox {:namespaces [neurojure.core neurojure.senses]
          :metadata {:doc/format :markdown}
          :source-uri "https://github.com/cguenthner/neurojure/blob/{git-commit}/{filepath}#L{line}"})
