# frozen_string_literal: true

require_relative "topical/version"

# Main module for topic modeling
module Topical
  class Error < StandardError; end
  
  # Autoload components for better performance
  autoload :Engine, "topical/engine"
  autoload :Topic, "topical/topic"
  autoload :Metrics, "topical/metrics"
  autoload :DimensionalityReducer, "topical/dimensionality_reducer"
  autoload :ModelSerializer, "topical/model_serializer"
  
  module Clustering
    autoload :Adapter, "topical/clustering/adapter"
    autoload :HDBSCANAdapter, "topical/clustering/hdbscan_adapter"
    autoload :KMeansAdapter, "topical/clustering/kmeans_adapter"
  end
  
  module Extractors
    autoload :TermExtractor, "topical/extractors/term_extractor"
  end
  
  module Labelers
    autoload :Base, "topical/labelers/base"
    autoload :TermBased, "topical/labelers/term_based"
    autoload :LLMBased, "topical/labelers/llm_based"
    autoload :Hybrid, "topical/labelers/hybrid"
  end
  
  # Convenience method for simple topic extraction
  # @param embeddings [Array<Array<Float>>] Document embeddings
  # @param documents [Array<String>] Document texts
  # @param options [Hash] Additional options
  # @return [Array<Topic>] Extracted topics
  def self.extract(embeddings:, documents:, **options)
    engine = Engine.new(**options)
    engine.fit(embeddings: embeddings, documents: documents)
  end
  
  # Check if red-candle is available for enhanced features
  def self.llm_available?
    @llm_available ||= begin
      require 'red-candle'
      true
    rescue LoadError
      false
    end
  end
end
