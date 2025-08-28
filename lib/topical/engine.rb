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
      
      puts "Starting topic extraction..." if @verbose
      
      # Step 1: Optionally reduce dimensions
      working_embeddings = @embeddings
      if @reduce_dimensions && @embeddings.first.length > @n_components
        puts "  Reducing dimensions from #{@embeddings.first.length} to #{@n_components}..." if @verbose
        working_embeddings = reduce_dimensions(@embeddings)
      end
      
      # Step 2: Cluster embeddings
      puts "  Clustering #{working_embeddings.length} documents..." if @verbose
      cluster_ids = @clustering_adapter.fit_predict(working_embeddings)
      
      # Step 3: Build topics from clusters
      puts "  Building topics from clusters..." if @verbose
      @topics = build_topics(cluster_ids)
      
      # Step 4: Extract terms for each topic
      puts "  Extracting distinctive terms..." if @verbose
      extract_topic_terms
      
      # Step 5: Generate labels
      puts "  Generating topic labels..." if @verbose
      generate_topic_labels
      
      if @verbose
        n_noise = cluster_ids.count(-1)
        puts "Found #{@topics.length} topics (plus #{n_noise} outliers)"
      end
      
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
    
    def reduce_dimensions(embeddings)
      # For now, just use UMAP from ClusterKit if available
      begin
        require 'clusterkit'
        
        # Adjust parameters based on data size
        n_samples = embeddings.size
        n_components = [@n_components, n_samples - 1, 50].min
        n_neighbors = [15, n_samples - 1].min
        
        umap = ClusterKit::Dimensionality::UMAP.new(
          n_components: n_components,
          n_neighbors: n_neighbors,
          random_seed: 42  # For reproducibility
        )
        
        umap.fit_transform(embeddings)
      rescue => e
        puts "Warning: Dimensionality reduction failed: #{e.message}" if @verbose
        embeddings  # Return original if reduction fails
      end
    end
    
    def build_topics(cluster_ids)
      # Group documents by cluster
      clusters = {}
      cluster_ids.each_with_index do |cluster_id, doc_idx|
        next if cluster_id == -1  # Skip outliers
        clusters[cluster_id] ||= []
        clusters[cluster_id] << doc_idx
      end
      
      # Create Topic objects
      clusters.map do |cluster_id, doc_indices|
        Topic.new(
          id: cluster_id,
          document_indices: doc_indices,
          documents: doc_indices.map { |i| @documents[i] },
          embeddings: doc_indices.map { |i| @embeddings[i] },
          metadata: doc_indices.map { |i| @metadata[i] }
        )
      end.sort_by(&:id)
    end
    
    def extract_topic_terms
      @topics.each do |topic|
        # Extract distinctive terms using c-TF-IDF
        terms = @term_extractor.extract_distinctive_terms(
          topic_docs: topic.documents,
          all_docs: @documents,
          top_n: 20
        )
        
        topic.terms = terms
      end
    end
    
    def generate_topic_labels
      @topics.each do |topic|
        topic.label = @labeler.generate_label(topic)
      end
    end
    
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