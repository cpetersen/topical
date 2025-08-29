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
      k: nil,  # Add k as explicit parameter
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
      @options[:k] = k if k  # Store k in options if provided
      
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
      @cluster_ids = @clustering_adapter.fit_predict(working_embeddings)
      
      # Step 3: Build topics from clusters
      puts "  Building topics from clusters..." if @verbose
      @topics = build_topics(@cluster_ids)
      
      # Step 4: Extract terms for each topic
      puts "  Extracting distinctive terms..." if @verbose
      extract_topic_terms
      
      # Step 5: Generate labels
      puts "  Generating topic labels..." if @verbose
      generate_topic_labels
      
      if @verbose
        n_noise = @cluster_ids.count(-1)
        puts "Found #{@topics.length} topics (plus #{n_noise} outliers)"
      end
      
      @topics
    end
    
    # Transform new documents using fitted model
    def transform(embeddings, documents: nil)
      raise "Must call fit before transform" if @topics.empty?
      
      # Use approximate prediction if available
      if @clustering_adapter.respond_to?(:approximate_predict)
        @clustering_adapter.approximate_predict(embeddings)
      else
        # Fallback: assign to nearest topic centroid
        assign_to_nearest_topic(embeddings)
      end
    end
    
    def get_topic(topic_id)
      @topics.find { |t| t.id == topic_id }
    end
    
    def outliers
      return [] unless @cluster_ids
      @documents.each_with_index.select { |_, idx| 
        @cluster_ids[idx] == -1 
      }.map(&:first)
    end
    
    # Save the model
    def save(path)
      require 'json'
      config = {
        clustering_method: @clustering_method,
        min_cluster_size: @min_cluster_size,
        min_samples: @min_samples,
        reduce_dimensions: @reduce_dimensions,
        n_components: @n_components,
        labeling_method: @labeling_method
      }
      
      # Include k for kmeans
      if @clustering_method == :kmeans
        config[:k] = @options[:k] || @topics.length
      end
      
      data = {
        topics: @topics.map(&:to_h),
        config: config
      }
      File.write(path, JSON.pretty_generate(data))
    end
    
    # Load a model
    def self.load(path)
      require 'json'
      data = JSON.parse(File.read(path), symbolize_names: true)
      
      # Make sure k is passed for kmeans and convert string keys to symbols
      config = data[:config]
      config[:clustering_method] = config[:clustering_method].to_sym if config[:clustering_method]
      config[:labeling_method] = config[:labeling_method].to_sym if config[:labeling_method]
      
      if config[:clustering_method] == :kmeans && !config[:k]
        # Extract k from saved topics or use default
        config[:k] = data[:topics]&.length || 5
      end
      
      engine = new(**config)
      # Reconstruct topics
      engine.instance_variable_set(:@topics, data[:topics].map { |t| Topic.from_h(t) })
      engine
    end
    
    private
    
    def reduce_dimensions(embeddings)
      begin
        require 'clusterkit'
        
        # Validate embeddings before UMAP
        valid_embeddings, invalid_indices = validate_embeddings_for_umap(embeddings)
        
        if valid_embeddings.empty?
          raise "No valid embeddings for dimensionality reduction. " \
                "All embeddings contain invalid values (NaN, Infinity, or non-numeric)."
        end
        
        if invalid_indices.any? && @verbose
          puts "  Warning: #{invalid_indices.size} embeddings with invalid values removed"
        end
        
        # Adjust parameters based on data size
        n_samples = valid_embeddings.size
        n_components = [@n_components, n_samples - 1, 50].min
        n_neighbors = [15, n_samples - 1].min
        
        if @verbose && n_components != @n_components
          puts "  Adjusted n_components to #{n_components} (was #{@n_components}) for #{n_samples} samples"
        end
        
        umap = ClusterKit::Dimensionality::UMAP.new(
          n_components: n_components,
          n_neighbors: n_neighbors,
          random_seed: 42
        )
        
        reduced = umap.fit_transform(valid_embeddings)
        
        # If we had to remove invalid embeddings, reconstruct the full array
        if invalid_indices.any?
          full_reduced = []
          valid_idx = 0
          embeddings.size.times do |i|
            if invalid_indices.include?(i)
              # Use zeros for invalid embeddings (they'll be outliers anyway)
              full_reduced << Array.new(n_components, 0.0)
            else
              full_reduced << reduced[valid_idx]
              valid_idx += 1
            end
          end
          full_reduced
        else
          reduced
        end
      rescue LoadError
        puts "Warning: Dimensionality reduction requires ClusterKit. Using original embeddings." if @verbose
        embeddings
      rescue => e
        puts "Warning: Dimensionality reduction failed: #{e.message}" if @verbose
        embeddings
      end
    end
    
    def validate_embeddings_for_umap(embeddings)
      valid = []
      invalid_indices = []
      
      embeddings.each_with_index do |embedding, idx|
        if embedding.is_a?(Array) && 
           embedding.all? { |v| v.is_a?(Numeric) && v.finite? }
          valid << embedding
        else
          invalid_indices << idx
        end
      end
      
      [valid, invalid_indices]
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
    
    def assign_to_nearest_topic(embeddings)
      # Simple nearest centroid assignment
      topic_centroids = @topics.map(&:centroid)
      
      embeddings.map do |embedding|
        distances = topic_centroids.map do |centroid|
          # Euclidean distance
          Math.sqrt(embedding.zip(centroid).map { |a, b| (a - b) ** 2 }.sum)
        end
        
        min_idx = distances.index(distances.min)
        @topics[min_idx].id
      end
    end
  end
end