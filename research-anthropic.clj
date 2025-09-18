#!/usr/bin/env bb
;; research.clj - Claude research CLI tool
;; Usage: ./research.clj --prompt "query" --output results.md
;;    or: ./research.clj --prompt-file prompt.md --output results.md

(ns research-anthropic
  (:require [babashka.http-client :as http]
            [cheshire.core :as json]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.tools.cli :refer [parse-opts]]))

(def anthropic-api-key (System/getenv "ANTHROPIC_API_KEY"))
(def serper-api-key (System/getenv "SERPER_API_KEY")) ; Optional: for web search

(defn search-web
  "Search web using Serper API (more cost-effective than Claude's native search)"
  [query]
  (if serper-api-key
    (let [response (http/post "https://google.serper.dev/search"
                              {:headers {"X-API-KEY" serper-api-key
                                         "Content-Type" "application/json"}
                               :body (json/generate-string {:q query})})]
      (when (= 200 (:status response))
        (let [data (json/parse-string (:body response) true)]
          (->> (concat (get data :organic [])
                       (get data :answerBox [])
                       (get data :knowledgeGraph []))
               (take 5)
               (map #(str "Title: " (:title %)
                          "\nSnippet: " (:snippet %)
                          "\nURL: " (:link %)))
               (str/join "\n\n")))))
    nil))

(defn claude-request
  "Make a request to Claude API"
  [messages & {:keys [with-search?]}]
  (let [tools (when with-search?
                [{:name "web_search"
                  :description "Search the web for information"
                  :input_schema {:type "object"
                                 :properties {:query {:type "string"}}
                                 :required ["query"]}}])
        request-body (cond-> {:model "claude-3-5-sonnet-20241022"
                              :max_tokens 4000
                              :messages messages}
                       with-search? (assoc :tools tools
                                           :tool_choice "auto"))
        response (http/post "https://api.anthropic.com/v1/messages"
                            {:headers {"x-api-key" anthropic-api-key
                                       "anthropic-version" "2023-06-01"
                                       "content-type" "application/json"}
                             :body (json/generate-string request-body)})]
    (when (= 200 (:status response))
      (let [data (json/parse-string (:body response) true)]
        (-> data :content first :text)))))

(defn research-with-external-search
  "Perform research using external search + Claude analysis"
  [query]
  (let [search-results (search-web query)
        system-prompt "You are an expert research assistant. Analyze the provided search results and create a comprehensive research report."
        user-message (if search-results
                       (format "Research Query: %s\n\nSearch Results:\n%s\n\nPlease analyze these results and provide a detailed research report."
                               query search-results)
                       (format "Research Query: %s\n\nPlease provide a comprehensive analysis based on your knowledge."
                               query))
        messages [{:role "user" :content user-message}]]
    (claude-request messages)))

(defn research-direct
  "Use Claude's native capabilities (expensive web search or knowledge base)"
  [query use-native-search?]
  (let [system-prompt "You are an expert research assistant. Provide comprehensive, well-structured research reports."
        messages [{:role "user"
                   :content (str "Please research the following topic thoroughly and provide a detailed report:\n\n" query)}]]
    (claude-request messages :with-search? use-native-search?)))

(def cli-options
  [["-p" "--prompt PROMPT" "Research prompt string"]
   ["-f" "--prompt-file FILE" "File containing research prompt"]
   ["-o" "--output FILE" "Output markdown file"
    :default "research-output.md"]
   ["-s" "--use-serper" "Use Serper API for web search (recommended)"
    :default true]
   ["-n" "--native-search" "Use Claude's native web search ($10/1000 searches)"
    :default false]
   ["-h" "--help"]])

(defn -main [& args]
  (let [{:keys [options errors summary]} (parse-opts args cli-options)]
    (cond
      (:help options)
      (do (println "Claude Research CLI Tool")
          (println summary)
          (System/exit 0))

      errors
      (do (println "Error:" (str/join "\n" errors))
          (System/exit 1))

      (not anthropic-api-key)
      (do (println "Error: ANTHROPIC_API_KEY environment variable not set")
          (System/exit 1))

      :else
      (let [prompt (cond
                     (:prompt options) (:prompt options)
                     (:prompt-file options) (slurp (:prompt-file options))
                     :else (do (println "Error: Either --prompt or --prompt-file required")
                               (System/exit 1)))
            _ (println "🔍 Starting research on:" (subs prompt 0 (min 50 (count prompt))) "...")
            result (cond
                     (and (:use-serper options) serper-api-key)
                     (do (println "📡 Using Serper API for web search...")
                         (research-with-external-search prompt))

                     (:native-search options)
                     (do (println "💰 Using Claude's native web search (expensive)...")
                         (research-direct prompt true))

                     :else
                     (do (println "🧠 Using Claude's knowledge base (no web search)...")
                         (research-direct prompt false)))]
        (if result
          (do (spit (:output options) result)
              (println "✅ Research complete! Results saved to:" (:output options)))
          (do (println "❌ Research failed. Check your API keys and network connection.")
              (System/exit 1)))))))

;; Run main if script is executed directly
(when (= *file* (first *command-line-args*))
  (apply -main (rest *command-line-args*)))