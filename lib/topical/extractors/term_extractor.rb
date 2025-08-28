# frozen_string_literal: true

require 'set'

module Topical
  module Extractors
    # Extracts distinctive terms from documents using c-TF-IDF
    class TermExtractor
      # Default English stop words
      DEFAULT_STOP_WORDS = Set.new(%w[
        the be to of and a in that have i it for not on with he as you do at
        this but his by from they we say her she or an will my one all would
        there their what so up out if about who get which go me when make can
        like time no just him know take people into year your good some could
        them see other than then now look only come its over think also back
        after use two how our work first well way even new want because any
        these give day most us is was are been has had were said did get may
      ])
      
      def initialize(stop_words: DEFAULT_STOP_WORDS, min_word_length: 3, max_word_length: 20)
        @stop_words = stop_words
        @min_word_length = min_word_length
        @max_word_length = max_word_length
      end
      
      # Extract distinctive terms using c-TF-IDF
      # @param topic_docs [Array<String>] Documents in the topic
      # @param all_docs [Array<String>] All documents in the corpus
      # @param top_n [Integer] Number of top terms to return
      # @return [Array<String>] Top distinctive terms
      def extract_distinctive_terms(topic_docs:, all_docs:, top_n: 20)
        # Tokenize and count terms in topic
        topic_terms = count_terms(topic_docs)
        
        # Tokenize and count document frequency across all docs
        doc_frequencies = compute_document_frequencies(all_docs)
        
        # Compute c-TF-IDF scores
        scores = {}
        total_docs = all_docs.length.to_f
        
        topic_terms.each do |term, tf|
          # c-TF-IDF formula: tf * log(N / df)
          df = doc_frequencies[term] || 1
          idf = Math.log(total_docs / df)
          scores[term] = tf * idf
        end
        
        # Return top scoring terms
        scores.sort_by { |_, score| -score }
               .first(top_n)
               .map(&:first)
      end
      
      private
      
      def tokenize(text)
        # Simple tokenization
        text.downcase
            .split(/\W+/)
            .select { |word| valid_word?(word) }
      end
      
      def valid_word?(word)
        word.length >= @min_word_length &&
        word.length <= @max_word_length &&
        !@stop_words.include?(word) &&
        !word.match?(/^\d+$/)  # Not pure numbers
      end
      
      def count_terms(documents)
        terms = Hash.new(0)
        
        documents.each do |doc|
          tokenize(doc).each do |word|
            terms[word] += 1
          end
        end
        
        terms
      end
      
      def compute_document_frequencies(documents)
        doc_frequencies = Hash.new(0)
        
        documents.each do |doc|
          # Use set to count each term once per document
          unique_terms = Set.new(tokenize(doc))
          unique_terms.each do |term|
            doc_frequencies[term] += 1
          end
        end
        
        doc_frequencies
      end
    end
  end
end