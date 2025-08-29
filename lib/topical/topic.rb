# frozen_string_literal: true

module Topical
  # Represents a discovered topic
  class Topic
    attr_reader :id, :document_indices, :documents, :embeddings, :metadata
    attr_accessor :terms, :label, :description, :distinctiveness
    attr_writer :coherence
    
    def initialize(id:, document_indices:, documents:, embeddings:, metadata: nil)
      @id = id
      @document_indices = document_indices
      @documents = documents
      @embeddings = embeddings
      @metadata = metadata || []
      @terms = []
      @label = nil
      @description = nil
      @coherence = nil
      @distinctiveness = 0.0
    end
    
    # Number of documents in this topic
    def size
      @documents.length
    end
    
    # Compute the centroid of the topic
    def centroid
      @centroid ||= compute_centroid
    end
    
    # Get the most representative documents
    # @param k [Integer] Number of documents to return
    # @return [Array<String>] Representative documents
    def representative_docs(k: 3)
      return @documents if @documents.length <= k
      
      # Find documents closest to centroid
      distances = @embeddings.map { |embedding| distance_to_centroid(embedding) }
      
      # Get indices of k smallest distances
      top_indices = distances.each_with_index.sort_by(&:first).first(k).map(&:last)
      top_indices.map { |i| @documents[i] }
    end
    
    # Compute topic coherence (simple PMI-based score)
    def coherence
      @coherence ||= compute_coherence
    end
    
    # Convert to hash for serialization
    def to_h
      {
        id: @id,
        label: @label,
        description: @description,
        size: size,
        terms: @terms,
        coherence: @coherence,
        distinctiveness: @distinctiveness,
        document_indices: @document_indices
      }
    end
    
    # Create from hash
    def self.from_h(hash)
      topic = new(
        id: hash[:id],
        document_indices: hash[:document_indices],
        documents: [],  # Would need to be reconstructed
        embeddings: []   # Would need to be reconstructed
      )
      topic.label = hash[:label]
      topic.description = hash[:description]
      topic.terms = hash[:terms]
      topic.coherence = hash[:coherence] || 0.0
      topic.distinctiveness = hash[:distinctiveness] || 0.0
      topic
    end
    
    private
    
    def compute_coherence
      # Use the Metrics module for proper coherence calculation
      return 0.0 if @terms.empty? || @documents.empty?
      
      Metrics.compute_coherence(@terms, @documents, top_n: 10)
    end
    
    def compute_centroid
      return [] if @embeddings.empty?
      
      # Compute mean of all embeddings
      dim = @embeddings.first.length
      centroid = Array.new(dim, 0.0)
      
      @embeddings.each do |embedding|
        embedding.each_with_index do |val, idx|
          centroid[idx] += val
        end
      end
      
      centroid.map { |val| val / @embeddings.length }
    end
    
    def distance_to_centroid(embedding)
      # Euclidean distance
      Math.sqrt(
        embedding.zip(centroid).map { |a, b| (a - b) ** 2 }.sum
      )
    end
  end
end