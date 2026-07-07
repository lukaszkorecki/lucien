(ns lucien.main
  "Prototype semantic file indexer."
  ;; TODO: add a logger
  (:require [babashka.fs :as fs])
  (:import
   [dev.langchain4j.data.embedding Embedding]
   [dev.langchain4j.model.embedding.onnx.allminilml6v2q AllMiniLmL6V2QuantizedEmbeddingModel]
   [dev.langchain4j.model.output Response] ;; search
   [org.apache.lucene.analysis.standard StandardAnalyzer]
   [org.apache.lucene.document
    Document
    Field$Store
    KnnFloatVectorField
    StoredField
    StringField]
   [org.apache.lucene.index
    DirectoryReader
    IndexWriter
    IndexWriterConfig
    IndexWriterConfig$OpenMode
    VectorSimilarityFunction]
   [org.apache.lucene.search IndexSearcher KnnFloatVectorQuery ScoreDoc]
   [org.apache.lucene.store FSDirectory]))

(set! *warn-on-reflection* true)

;; Embeddings
;; Model construction is expensive (loads the ONNX graph + tokenizer) and the
;; instance is thread-safe, so build once and reuse. `delay` so requiring the ns
;; is cheap and the model only loads on first `embed`.

(defonce ^:private model
  (delay (AllMiniLmL6V2QuantizedEmbeddingModel.)))

(defn embed
  "String -> 384-element float[]. First call is slow (model init + JIT warmup)."
  ^floats [^String s]
  (let [response (.embed ^AllMiniLmL6V2QuantizedEmbeddingModel @model s)
        content (Response/.content response)]
    (Embedding/.vector content)))

;; Files -> chunks
;; Dumb fixed-window chunker. Fine for wiring.
;; TODO: Swap for commonmark-java or treesitter to get better semantic structure?

(defn chunk-text [^String s ^long size ^long overlap]
  (let [step (max 1 (- size overlap))
        n (count s)]
    (loop [start 0
           acc []]
      (if (>= start n)
        acc
        (let [end (min n (+ start size))
              piece (subs s start end)]
          (if (>= end n)
            (conj acc piece)
            (recur (+ start step) (conj acc piece))))))))

(defn text-files
  "Seq of java.nio.file.Path for the extensions we care about, recursively."
  [dir]
  (->> (fs/glob dir "**/*.{md,markdown,txt}")
       (filter fs/regular-file?)))

;; ---------------------------------------------------------------------------
;; Build index (on disk)
;; ---------------------------------------------------------------------------

(defn build-index!
  "Walk src-dir, embed each chunk, write vectors + text to a Lucene index at index-dir.
   OpenMode/CREATE wipes any existing index — so this is a full rebuild each call."
  [src-dir index-dir]
  (let [indexer-config (doto (IndexWriterConfig. (StandardAnalyzer.))
                         ;; TODO: use CREATE_OR_APPEND and generate IDs for docs + hashing
                         ;; to make reindexing work, right now this is full rebuild
                         (.setOpenMode IndexWriterConfig$OpenMode/CREATE))]
    (with-open [dir (FSDirectory/open (fs/path index-dir))

                writer (IndexWriter. dir indexer-config)]
      (doseq [f (text-files src-dir)
              :let [path-str (str f)
                    content (slurp (fs/file f))]
              [i chunk] (map-indexed vector (chunk-text content 1000 100))]
        (let [v (embed chunk)
              doc (doto (Document.)
                    (Document/.add (StringField. "path" path-str Field$Store/YES))
                    (Document/.add (StoredField. "chunk" (int i)))
                    (Document/.add (StoredField. "text" ^String chunk))
                    ;; COSINE handles normalization internally — safest with raw model output.
                    (Document/.add (KnnFloatVectorField. "vector" v VectorSimilarityFunction/COSINE)))]
          (.addDocument writer doc)))
      (.commit writer)
      :ok)))

;; ---------------------------------------------------------------------------
;; Search
;; ---------------------------------------------------------------------------

(defn search
  "Natural-language query -> top-k chunks as maps {:score :path :text}.
   Opens a fresh reader each call (fine for the REPL; cache it when you productionize)."
  [index-dir ^String query k]
  (with-open [dir (FSDirectory/open (fs/path index-dir))
              reader (DirectoryReader/open dir)]
    (let [searcher (IndexSearcher. reader)
          qvec (embed query)
          q (KnnFloatVectorQuery. "vector" qvec (int k))
          top (.search searcher q (int k))
          sf (.storedFields searcher)] ;; Lucene 10: NOT searcher.doc(id)
      ;; mapv to realize results *before* the reader closes.
      (mapv (fn [sd]
              (let [d ^Document (.document sf (.-doc ^ScoreDoc sd))]
                {:score (.-score ^ScoreDoc sd)
                 :path (.get d "path")
                 :text (.get d "text")}))
            (.scoreDocs top)))))

(defn index-stats [index-dir]
  (with-open [dir (FSDirectory/open (fs/path index-dir))
              reader (DirectoryReader/open dir)]
    (.numDocs reader)))

(comment
  ;; --- REPL session ---
  (build-index! "/Users/lukasz/notes" ".index")

  (search ".index" "how do I configure rclone bisync on launchd" 5)

  ;; warm the model so the first real query isn't slow
  (embed "warmup")

  ;; sanity-check what got indexed
  (with-open [dir (FSDirectory/open (fs/path ".index"))
              reader (DirectoryReader/open dir)]
    (.numDocs reader)))

(defn -main [& [command index-dir query-or-src-dir]]
  (case command
    "build-index" (build-index! (str (fs/absolutize query-or-src-dir)) (str (fs/absolutize index-dir)))
    "search" (->> (search (str (fs/absolutize index-dir)) query-or-src-dir 5)
                  (mapv (fn [{:keys [score path text]}]
                          (printf "> %s\n: %s | %s\n"
                                  text
                                  score path))))

    (do
      (println "oh no")
      (System/exit 1))))
