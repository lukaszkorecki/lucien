(ns lucien.main
  "Content indexer + search skeleton.
   Plan: LSP-aware source indexing + nat-lang search over docs/code."
  (:require [babashka.fs :as fs])
  (:import
   [dev.langchain4j.data.embedding Embedding]
   [dev.langchain4j.model.embedding.onnx.allminilml6v2q AllMiniLmL6V2QuantizedEmbeddingModel]
   [dev.langchain4j.model.output Response]
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
    Term
    VectorSimilarityFunction]
   [org.apache.lucene.search
    IndexSearcher
    KnnFloatVectorQuery
    ScoreDoc
    SearcherFactory
    SearcherManager]
   [org.apache.lucene.store FSDirectory]))

(set! *warn-on-reflection* true)

;; ---------------------------------------------------------------------------
;; Embeddings
;; ---------------------------------------------------------------------------

;; Model construction is expensive (loads the ONNX graph + tokenizer) and the
;; instance is thread-safe, so build once and reuse.
(defonce ^:private model
  (delay (AllMiniLmL6V2QuantizedEmbeddingModel.)))

(defn embed
  "Embed s into a 384-element float vector using the all-MiniLM-L6-v2 model.
   The model is loaded lazily on the first call, which is slow (~1-2s); subsequent
   calls are fast. The model instance is thread-safe and reused across calls."
  ^floats [^String s]
  (let [response (.embed ^AllMiniLmL6V2QuantizedEmbeddingModel @model s)
        content (Response/.content response)]
    (Embedding/.vector content)))

;; ---------------------------------------------------------------------------
;; Chunking
;; ---------------------------------------------------------------------------

(defn chunk-text
  "Split s into a vector of overlapping fixed-size substrings.

   size    — character count of each chunk.
   overlap — how many characters the end of one chunk shares with the start
             of the next. Must be less than size.

   The step between chunk start positions is (size - overlap), so consecutive
   chunks share `overlap` characters. The final chunk is whatever remains and
   may be shorter than size. Example with size=10 overlap=3:

     s = \"abcdefghijklmno\"
     chunks: [\"abcdefghij\" \"hijklmnopq\" ...]  (step = 7)

   Overlap exists so that a sentence or code expression split across a chunk
   boundary still appears whole in at least one chunk, improving recall for
   vector search."
  [^String s ^long size ^long overlap]
  (let [step (max 1 (- size overlap))
        n (count s)]
    (loop [start 0 acc []]
      (if (>= start n)
        acc
        (let [end (min n (+ start size))
              piece (subs s start end)]
          (if (>= end n)
            (conj acc piece)
            (recur (+ start step) (conj acc piece))))))))

;; ---------------------------------------------------------------------------
;; File discovery
;; ---------------------------------------------------------------------------

(defn text-files
  "Return a seq of java.nio.file.Path for all .md, .markdown, and .txt files
   found recursively under dir. Symlinks and non-regular files are excluded."
  [dir]
  (->> (fs/glob dir "**/*.{md,markdown,txt}")
       (filter fs/regular-file?)))

;; ---------------------------------------------------------------------------
;; Document construction
;; ---------------------------------------------------------------------------

(defn- doc-for
  "Build a Lucene Document representing one chunk of a file.
   path-str  — absolute path stored and returned in search results.
   chunk-idx — zero-based position of this chunk within the file (stored for debugging).
   text      — the raw chunk text, stored and returned in search results.
   vec       — the embedding vector, indexed as a KNN float vector field."
  ^Document [^String path-str ^long chunk-idx ^String text ^floats vec]
  (doto (Document.)
    (Document/.add (StringField. "path" path-str Field$Store/YES))
    (Document/.add (StoredField. "chunk" (int chunk-idx)))
    (Document/.add (StoredField. "text" text))
    ;; COSINE handles normalization internally — safest with raw model output.
    (Document/.add (KnnFloatVectorField. "vector" vec VectorSimilarityFunction/COSINE))))

;; ---------------------------------------------------------------------------
;; Index writer lifecycle
;; ---------------------------------------------------------------------------

(defn open-writer!
  "Open an IndexWriter against index-dir in CREATE_OR_APPEND mode.
   Use this for the long-lived writer held by the server component.
   Caller is responsible for closing (implements Closeable, so with-open works)."
  ^IndexWriter [index-dir]
  (let [dir (FSDirectory/open (fs/path index-dir))
        cfg (doto (IndexWriterConfig. (StandardAnalyzer.))
              (.setOpenMode IndexWriterConfig$OpenMode/CREATE_OR_APPEND))]
    (IndexWriter. dir cfg)))

;; ---------------------------------------------------------------------------
;; Full index rebuild
;; ---------------------------------------------------------------------------

(defn build-index!
  "Embed every text file under src-dir and write a fresh Lucene index at index-dir.
   Uses CREATE mode, so any existing index is wiped. Intended for the initial
   build or a forced full rebuild; for ongoing updates use reindex-file! instead."
  [src-dir index-dir]
  (let [cfg (doto (IndexWriterConfig. (StandardAnalyzer.))
              (.setOpenMode IndexWriterConfig$OpenMode/CREATE))]
    (with-open [dir (FSDirectory/open (fs/path index-dir))
                writer (IndexWriter. dir cfg)]
      (doseq [f (text-files src-dir)
              :let [path-str (str f)
                    content (slurp (fs/file f))]
              [i chunk] (map-indexed vector (chunk-text content 1000 100))]
        (.addDocument writer (doc-for path-str i chunk (embed chunk))))
      (.commit writer)
      :ok)))

;; ---------------------------------------------------------------------------
;; Incremental reindex
;; ---------------------------------------------------------------------------

(defn reindex-file!
  "Delete all existing chunks for f from the index, then re-chunk, re-embed,
   and re-add them. writer must already be open. Changes are not visible to
   searchers until (.maybeRefresh sm) is called — batch multiple reindexes
   before refreshing to avoid multiple reader flips."
  [^IndexWriter writer f]
  (let [path-str (str f)
        terms ^Term/1 (into-array Term [(Term. "path" path-str)])]
    (.deleteDocuments writer terms)
    (doseq [[i chunk] (map-indexed vector (chunk-text (slurp (fs/file f)) 1000 100))]
      (.addDocument writer (doc-for path-str i chunk (embed chunk))))))

(defn on-fs-events!
  "Process a batch of filesystem events against the open writer, then flip the
   searcher to make the changes visible. deleted-paths are removed from the index;
   changed-paths are fully reindexed. The SearcherManager refresh happens once at
   the end so all changes in the batch become visible atomically."
  [^IndexWriter writer ^SearcherManager sm changed-paths deleted-paths]
  (doseq [p deleted-paths]
    (let [terms ^Term/1 (into-array Term [(Term. "path" (str p))])]
      (.deleteDocuments writer terms)))
  (doseq [p changed-paths]
    (reindex-file! writer p))
  ;; Make the whole batch visible in one shot.
  (.maybeRefresh sm))

;; ---------------------------------------------------------------------------
;; SearcherManager lifecycle
;; ---------------------------------------------------------------------------

(defn open-searcher-manager!
  "Open a SearcherManager backed by index-dir. The manager holds a shared,
   ref-counted IndexSearcher that is cheap to acquire per query. Caller must
   close it when done (implements Closeable, so with-open works).
   The index at index-dir must already exist (run build-index! first)."
  ^SearcherManager [index-dir]
  (SearcherManager.
   (FSDirectory/open (fs/path index-dir))
   (SearcherFactory.)))

;; ---------------------------------------------------------------------------
;; Search
;; ---------------------------------------------------------------------------

(defn search
  "Embed query and return the top-k nearest chunks as maps of {:score :path :text}.
   Acquires an IndexSearcher from sm for the duration of the call and releases it
   on exit — even on exception. sm is never closed here; that is the caller's job."
  [^SearcherManager sm ^String query k]
  (let [^IndexSearcher s (.acquire sm)]
    (try
      (let [qvec (embed query)
            top (.search s (KnnFloatVectorQuery. "vector" qvec (int k)) (int k))
            sf (.storedFields s)]
        (mapv (fn [sd]
                (let [d ^Document (.document sf (.-doc ^ScoreDoc sd))]
                  {:score (.-score ^ScoreDoc sd)
                   :path (.get d "path")
                   :text (.get d "text")}))
              (.scoreDocs top)))
      (finally
        (.release sm s)))))

;; ---------------------------------------------------------------------------
;; Diagnostics
;; ---------------------------------------------------------------------------

(defn index-stats
  "Return the number of documents currently stored in the index at index-dir.
   Opens and closes a DirectoryReader; not intended to be called on a hot path."
  [index-dir]
  (with-open [dir (FSDirectory/open (fs/path index-dir))
              reader (DirectoryReader/open dir)]
    (.numDocs reader)))

;; ---------------------------------------------------------------------------
;; CLI entry point
;; ---------------------------------------------------------------------------

(defn -main
  "CLI entry point.
     build-index <index-dir> <src-dir>  — full rebuild of the index from src-dir.
     search      <index-dir> <query>    — print top-5 results for query."
  [& [command index-dir query-or-src-dir]]
  (case command
    "build-index"
    (do
      (build-index! (str (fs/absolutize query-or-src-dir))
                    (str (fs/absolutize index-dir)))

      (index-stats (str (fs/absolutize index-dir))))

    "search"
    (with-open [sm (open-searcher-manager! (str (fs/absolutize index-dir)))]
      (->> (search sm query-or-src-dir 5)
           (mapv (fn [{:keys [score path text]}]
                   (printf "> %s\n: %s | %s\n" text score path)))))

    (do
      (println "Usage: lucien <build-index|search> <index-dir> <src-dir|query>")
      (System/exit 1))))

(comment
  ;; REPL session
  (build-index! "/Users/lukasz/notes" ".index")
  (index-stats ".index")
  (embed "warmup")

  ;; Long-lived SM for interactive exploration
  (def sm (open-searcher-manager! ".index"))
  (search sm "how do I configure rclone bisync on launchd" 5)
  (.close sm))
