#!/usr/bin/env bb
(ns research
  (:require [babashka.cli :as cli]
            [babashka.curl :as curl]
            [cheshire.core :as json]
            [clojure.string :as str]
            [clojure.java.io :as io])
  (:import (java.time Instant)))

(def spec
  {:prompt       {:coerce :string}
   :prompt-file  {:coerce :string}
   :output       {:coerce :string :require true}
   :model        {:coerce :string :default "o4-mini-deep-research"}
   :no-web       {:coerce :boolean :default false}
   :help         {:coerce :boolean}})

(defn usage []
  (str "Deep Research CLI (babashka)\n\n"
       "Required env: OPENAI_API_KEY\n\n"
       "Options:\n"
       "  --prompt <text>\n"
       "  --prompt-file <file>\n"
       "  --output <file>\n"
       "  --model <name>            (default: o4-mini-deep-research)\n"
       "  --no-web                  (disable web_search tool)\n"
       "  -h, --help\n\n"
       "Examples:\n"
       "  bb research.clj --prompt \"EU HBM supply chain 2025\" --output out.md\n"
       "  bb research.clj --prompt-file prompt.md --output out.md\n"))

(defn read-prompt [{:keys [prompt prompt-file]}]
  (cond
    (and prompt prompt-file)
    (throw (ex-info "Use either --prompt or --prompt-file, not both." {}))
    prompt (str/trim prompt)
    prompt-file (-> prompt-file io/file slurp str/trim)
    :else (throw (ex-info "Provide --prompt or --prompt-file." {}))))

(defn extract-output-text [resp]
  (or (get resp "output_text")
      (let [outs (for [item (get resp "output")
                       :when (= "message" (get item "type"))
                       c (get item "content")
                       :when (= "output_text" (get c "type"))]
                   (get c "text"))]
        (str/trim (str/join "\n" outs)))))

(defn fail!
  ([msg] (binding [*out* *err*] (println msg)) (System/exit 2))
  ([msg m] (binding [*out* *err*]
             (println msg)
             (when m (println (with-out-str (clojure.pprint/pprint m)))))
           (System/exit 2)))

(defn -main [& args]
  (let [{:keys [opts arguments]} (cli/parse-args args {:spec spec
                                                       :exec-arg0 "research.clj"
                                                       :strict true})
        opts (cond-> opts (:help opts) (assoc :help true))]
    (when (or (:help opts) (seq arguments))
      (println (usage)) (System/exit 0))

    (let [api-key (System/getenv "OPENAI_API_KEY")]
      (when (str/blank? api-key)
        (fail! "OPENAI_API_KEY is not set.")))

    (let [prompt (try (read-prompt opts)
                      (catch Exception e (fail! (.getMessage e))))
          tools  (if (:no-web opts) [] [{:type "web_search"}])
          system "You are a deep research agent. Produce a concise, citation-rich Markdown report with sections and a Sources list of direct URLs. Prefer primary/official sources. State uncertainties explicitly."
          body   {:model (:model opts)
                  :input [{:role "system"
                           :content [{:type "input_text" :text system}]}
                          {:role "user"
                           :content [{:type "input_text" :text prompt}]}]
                  :tools tools}
          resp   (curl/post "https://api.openai.com/v1/responses"
                            {:headers {"Authorization" (str "Bearer " (System/getenv "OPENAI_API_KEY"))
                                       "Content-Type"  "application/json"}
                             :throw false
                             :body    (json/encode body)})]
      ;; Basic HTTP error handling
      (when (or (nil? (:status resp))
                (>= (:status resp) 300))
        (let [body-text (:body resp)
              parsed    (try (json/decode body-text) (catch Exception _ body-text))]
          (fail! (str "API error (status " (:status resp) "):") parsed)))

      (let [parsed (json/decode (:body resp))
            out    (or (extract-output-text parsed)
                       (json/encode parsed {:pretty true}))
            header (format "<!-- Generated: %s | model=%s -->\n"
                           (str (Instant/now)) (:model opts))]
        (spit (:output opts) (str header (str/trim out) "\n"))
        (println "Wrote" (:output opts))))))

(apply -main *command-line-args*)
