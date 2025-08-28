# frozen_string_literal: true

module Topical
  # Main engine for topic modeling
  class Engine
    attr_reader :topics, :clustering_adapter, :term_extractor, :labeler
    
    def initialize(
      clustering_method: :hdbscan,
      min_cluster_size: 5,
      min_samples: 3,
      reduce_dimensions: true,
      n_components: 50,
      labeling_method: :hybrid,
      llm_provider: nil,
      verbose: false,
      **options
    )
      @clustering_method = clustering_method
      @min_cluster_size = min_cluster_size
      @min_samples = min_samples
      @reduce_dimensions = reduce_dimensions
      @n_components = n_components
      @labeling_method = labeling_method
      @llm_provider = llm_provider
      @verbose = verbose
      @options = options
      
      @clustering_adapter = build_clustering_adapter
      @term_extractor = Extractors::TermExtractor.new
      @labeler = build_labeler
      @topics = []
    end
    
    # Fit the model to embeddings and documents
    # @param embeddings [Array<Array<Float>>] Document embeddings
    # @param documents [Array<String>] Document texts
    # @param metadata [Array<Hash>] Optional metadata for each document
    # @return [Array<Topic>] Extracted topics
    def fit(embeddings, documents, metadata: nil)
      raise ArgumentError, "Embeddings and documents must have same length" unless embeddings.length == documents.length
      
      @embeddings = embeddings
      @documents = documents
      @metadata = metadata || Array.new(documents.length) { {} }
      
      # TODO: Implement the full pipeline
      # Step 1: Optionally reduce dimensions
      # Step 2: Cluster embeddings
      # Step 3: Build topics from clusters
      # Step 4: Extract terms for each topic
      # Step 5: Generate labels
      
      @topics
    end
    
    # Transform new documents using fitted model
    def transform(embeddings, documents)
      raise "Must call fit before transform" if @topics.empty?
      # TODO: Implement transform
    end
    
    # Save the model
    def save(path)
      # TODO: Implement save
    end
    
    # Load a model
    def self.load(path)
      # TODO: Implement load
    end
    
    private
    
    def build_clustering_adapter
      case @clustering_method
      when :hdbscan
        Clustering::HDBSCANAdapter.new(
          min_cluster_size: @min_cluster_size,
          min_samples: @min_samples
        )
      when :kmeans
        Clustering::KMeansAdapter.new(k: @options[:k] || 5)
      else
        raise ArgumentError, "Unknown clustering method: #{@clustering_method}"
      end
    end
    
    def build_labeler
      case @labeling_method
      when :term_based
        Labelers::TermBased.new
      when :llm_based
        Labelers::LLMBased.new(provider: @llm_provider)
      when :hybrid
        Labelers::Hybrid.new(provider: @llm_provider)
      else
        Labelers::TermBased.new  # Default fallback
      end
    end
  end
end