(ns neurojure.senses
  "Utilities for data pre-processing."
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [neurojure.utils :as u]
            [tensure.core :as m]))

; TODO: Improve this so it can handle redirects if a data URL moves and provides better error messages
; for different HTTP status codes.
; TODO: Support some validation of the downloaded file (hash? size?)
(defn fetch-data
  "Given `k` (a keyword or seq of keywords to identify a dataset), either loads the dataset from an on-disk
  cache or else fetches it from `url` (a string). The cache location can be set using the
  `NEUROJURE_DATA_PATH` environment variable. Returns the path to the data file as a string."
  [k url]
  (let [root (or (System/getenv "NEUROJURE_DATA_PATH") "data")
        output-path (->> (cond (sequential? k) k
                               (integer? k) [(str k)]
                               (or (keyword? k) (string? k)) [k]
                               :else (u/throw-str "Invalid data key: '" k "'. "
                                                  "Expected a keyword or a seq of keywords"))
                         (map (fn [k]
                                (let [s (if (keyword? k)
                                          (name k)
                                          (str k))]
                                  (if (re-find #"^[a-zA-Z0-9\-\_]+$" s)
                                    s
                                    (u/throw-str "Invlaid data key element: '" k "'. Expected a keyword "
                                                 "with only letters, digis, '-', and '_'.")))))
                         (#(string/join "/" %))
                         (u/path-join root))
        output-file (io/file output-path)]
    (when-not (.exists output-file)
      (print (str "Fetching " url "..."))
      (io/make-parents output-file)
      (with-open [in (io/input-stream url)
                  out (io/output-stream output-file)]
        (io/copy in out))
      (println "Done."))
    output-path))

(defn rgba->int
  "Converts RGB, RGBA, or grayscale byte values (between 0-255) into an RGBA integer for use with a Java
  BufferedImage/TYPE_INT_RGBA."
  ([gray]
   (rgba->int gray gray gray))
  ([r g b]
   (rgba->int r g b 255))
  ([r g b a]
   (->> (bit-or (bit-shift-left (short a) 24)
                (bit-shift-left (short r) 16)
                (bit-shift-left (short g) 8)
                (short b))
        unchecked-int)))

(defn make-rgba-image
  "Creates an RGBA BufferedImage of the given width and height."
  [w h]
  (java.awt.image.BufferedImage. w h java.awt.image.BufferedImage/TYPE_INT_ARGB))

(defn data->image
  "Given a neurojure image representation (a tensor of shape [height, width, channels]), where channels are
  [R G B A], [R G B], or [Gray], returns a Java `BufferedImage` containing the data."
  [data]
  (let [[h w] (m/shape data)
        img (make-rgba-image w h)]
    (doseq [i (range h)
            j (range w)]
      (->> (m/select-range data i j :all)
           m/eseq
           (apply rgba->int)
           (.setRGB img i j)))
    img))

(defn data->image-array
  "Given a neurojure image set representation (a tensor of shape [image_count, height, width, channels]),
  where channels are [R G B A], [R G B], or [Gray], returns a Java `BufferedImage` containing the data
  arranged in `rows` and `cols` with an optional `margin`. The default is to tile the images to be
  approximately square (with respect to tile count) with 0 margin."
  ([data]
   (let [image-count (first (m/shape data))
         cols (Math/ceil (Math/sqrt image-count))
         rows (Math/ceil (/ image-count cols))]
     (data->image-array data rows cols)))
  ([data rows cols]
   (data->image-array data rows cols 0))
  ([data rows cols margin]
   (let [[img-count img-height img-width] (m/shape data)
         w-margin (+ img-width margin)
         h-margin (+ img-height margin)
         img (make-rgba-image (+ margin (* w-margin (inc cols)))
                              (+ margin (* h-margin (inc rows))))
         img-graphics (.getGraphics img)]
     (doseq [[i tile-data] (map-indexed vector (m/slices data))
             :let [row (Math/floor (/ i cols))
                   col (mod i cols)
                   ;_ (println row col)
                   tile-img (data->image tile-data)]]
       (.drawImage img-graphics tile-img
                   (int (+ (* col w-margin) margin))
                   (int (+ (* row h-margin) margin))
                   nil))
     img)))

(defn open-image
  "Writes the given `java.awt.image.BufferedImage` to a temporary PNG file and opens it using the default
  system viewer."
  [img]
  (let [tmp-file (java.io.File/createTempFile (u/uuid) ".png")]
    (try
      (.deleteOnExit tmp-file)
      (javax.imageio.ImageIO/write img "png" tmp-file)
      (.open (java.awt.Desktop/getDesktop) tmp-file)
      (catch Exception e
        (.delete tmp-file)))))

(defn make-one-hot-code
  "Given `categories`, a vector of objects whose identity can be determined by their hashes (keywords,
  strings, numbers, clojure collecitons, etc.), returns an object like
  `{:encode (fn [data] ...) :decode (fn [encoded-data] ...)}`. The `:encode` function takes a vector of
  objects and returns a matrix of shape [data-size category-count] representing `data` encoded in one-hot
  form. The `:decode` function takes a matrix like that returned by the `:encode` function and returns a
  vector like `categories`. Options include:

    - `:other?` (optional, default false) - a boolean indicating whether to include an additional category
      for any sample that does not exist in `categories`. If false, the encode function  will throw an
      Exception if it encounters a category not in `categories`. If true, the decode will return
     `:other-sentinel` for any  object not in `categories`.
    - `:other-sentinel` (optional, default -1) - any hashable (and non-nil) object that should be used as
      the category for any object not in `categories`."
  ([categories]
   (make-one-hot-code categories {}))
  ([categories options]
   (u/check-key-validity options [] [:other? :other-sentinel])
   (let [include-other? (:other? options)
         other-sentinel (or (:other-sentinel options) -1)
         distinct-categories (-> (if include-other?
                                   (cons other-sentinel categories)
                                   categories)
                                 distinct
                                 vec)
         category-count (count distinct-categories)
         category->hot-index (->> distinct-categories
                                  (map-indexed (fn [i category]
                                                 [category i]))
                                  (into {}))
         not-found-index (when include-other?
                           0)
         encode (fn [data]
                  (let [result (m/zeros [(count data) category-count])]
                    (doseq [[i el] (map-indexed #(vector %1 %2) data)]
                      (if-let [hot-index (get category->hot-index el not-found-index)]
                        (m/mset! result i hot-index 1)
                        (u/throw-str "Attempt to one-hot-encode an unknown category, '" el "'. "
                                     "`make-one-hot-code` must receive a vector of all possible "
                                     "categories, or else `:other?` must be set to `true` in "
                                     "options.")))
                    result))
         decode (fn [encoded-data]
                  (when-not (and (m/matrix? encoded-data)
                                 (= (m/dimension-count encoded-data 1) category-count))
                    (u/throw-str "Cannot decode a one-hot matrix of shape '" (m/shape encoded-data) "' with "
                                 "a decoder constructed for '" category-count "' categories. The number of "
                                 "columns in the encoded data matrix must equal the number of categories."))
                  (->> (m/argmax-along encoded-data 1)
                       m/eseq
                       (map (comp distinct-categories int))))]
     {:encode encode :decode decode})))

(def load-gloves
  "Given a vocab size in thousands (400, 1200, 1900, or 2200) and a vector size, returns a map of word
  to vector representation. Available vector sizes for the different vocabularies are at
  https://nlp.stanford.edu/projects/glove/."
  (memoize
    (fn
      [vocab-size vector-size]
      (let [; Map of vecab size to info.
            available {; Wikipedia 2014 + Gigawords
                       400 {:url "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
                            ; Map of vector size to corresponding zip file entry.
                            :entries {50 "glove.6B.50d.txt"
                                      100 "glove.6B.100d.txt"
                                      200 "glove.6B.200d.txt"
                                      300 "glove.6B.200d.txt"}}
                       ; Common Crawl 42B tokens
                       1900 {:url "http://downloads.cs.stanford.edu/nlp/data/glove.42B.300d.zip"
                             :entries {300 "glove.42B.300d.txt"}}
                       ; Common Crawl 840B tokens
                       2200 {:url "http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip"
                             :entries {300 "glove.840B.300d.zip"}}
                       ; Twitter
                       1200 {:url "http://downloads.cs.stanford.edu/nlp/data/glove.twitter.27B.zip"
                             :entires {25 "glove.twitter.27B.25d.txt"
                                       50 "glove.twitter.27B.50d.txt"
                                       100 "glove.twitter.27B.100d.txt"
                                       200 "glove.twitter.27B.200d.txt"}}}
            _ (when-not (get-in available [vocab-size :entries vector-size])
                (u/throw-str "Invalid vocabulary and GloVe size: '" vocab-size "'K words and '"
                             vector-size "'d vectors."))
            zip-file-path (fetch-data [:glove vocab-size] (get-in available [vocab-size :url]))
            zip-file (java.util.zip.ZipFile. "data/glove/400")
            zip-entry (.getEntry zip-file (get-in available [vocab-size :entries vector-size]))]
        (with-open [in (.getInputStream zip-file zip-entry)
                    rdr (io/reader in)]
          (->> (line-seq rdr)
               (pmap (fn [line]
                       (let [tokens (string/split line #" ")
                             word (first tokens)
                             word-vec (->> (rest tokens)
                                           (map #(Float/parseFloat %))
                                           m/array)]
                         [word word-vec])))
               (into {})))))))

(defn tokenize
  "Given a string, returns a seq of strings representing individual 'tokens'. Uses a very simple tokenization
  scheme whereby any substring is a separate token if it is 1) flanked by whitespace, or 2) is a
  non-alphabetical character that's not a '.' or ',' in a number."
  [s]
  (->> (re-seq #"\s*([a-zA-Z]+|[\d\.\,]+|[^\s\da-zA-Z]+)\s*" s)
       (map (comp string/lower-case second))))
