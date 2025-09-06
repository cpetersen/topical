# frozen_string_literal: true

module Topical
  # Handles saving and loading of topic models
  class ModelSerializer
    # Save a topic model to JSON file
    # @param engine [Engine] The engine instance to save
    # @param path [String] File path to save to
    def self.save(engine, path)
      require 'json'
      
      config = {
        clustering_method: engine.instance_variable_get(:@clustering_method),
        min_cluster_size: engine.instance_variable_get(:@min_cluster_size),
        min_samples: engine.instance_variable_get(:@min_samples),
        reduce_dimensions: engine.instance_variable_get(:@reduce_dimensions),
        n_components: engine.instance_variable_get(:@n_components),
        labeling_method: engine.instance_variable_get(:@labeling_method)
      }
      
      # Include k for kmeans
      options = engine.instance_variable_get(:@options)
      if config[:clustering_method] == :kmeans
        config[:k] = options[:k] || engine.topics.length
      end
      
      data = {
        topics: engine.topics.map(&:to_h),
        config: config
      }
      
      File.write(path, JSON.pretty_generate(data))
    end
    
    # Load a topic model from JSON file
    # @param path [String] File path to load from
    # @return [Engine] Loaded engine instance
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
      
      engine = Engine.new(**config)
      # Reconstruct topics
      engine.instance_variable_set(:@topics, data[:topics].map { |t| Topic.from_h(t) })
      engine
    end
  end
end