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
      # Simple coherence score based on term co-occurrence
      # Returns a value between 0 and 1
      return 0.0 if @terms.empty? || @documents.empty?
      
      # For now, return a simple heuristic based on term frequency
      # A more sophisticated implementation would use PMI or NPMI
      term_count = @terms.length
      doc_count = @documents.length
      
      # Basic score: more terms and more documents = better topic
      score = Math.log(term_count + 1) * Math.log(doc_count + 1) / 10.0
      [score, 1.0].min  # Cap at 1.0
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