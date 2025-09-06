# frozen_string_literal: true

require 'logger'

module Topical
  # Handles dimensionality reduction for embeddings using UMAP
  class DimensionalityReducer
    def initialize(n_components: 50, logger: nil)
      @n_components = n_components
      @logger = logger || Logger.new(IO::NULL, level: Logger::FATAL)
    end
    
    # Reduce dimensionality of embeddings if needed
    # @param embeddings [Array<Array<Float>>] Input embeddings
    # @return [Array<Array<Float>>] Reduced embeddings
    def reduce(embeddings)
      return embeddings if embeddings.empty?
      return embeddings if embeddings.first.length <= @n_components
      
      begin
        require 'clusterkit'
        
        # Validate embeddings before UMAP
        valid_embeddings, invalid_indices = validate_embeddings_for_umap(embeddings)
        
        if valid_embeddings.empty?
          raise "No valid embeddings for dimensionality reduction. " \
                "All embeddings contain invalid values (NaN, Infinity, or non-numeric)."
        end
        
        if invalid_indices.any?
          @logger.warn "  Warning: #{invalid_indices.size} embeddings with invalid values removed"
        end
        
        # Adjust parameters based on data size
        n_samples = valid_embeddings.size
        n_components = [@n_components, n_samples - 1, 50].min
        n_neighbors = [15, n_samples - 1].min
        
        if n_components != @n_components
          @logger.info "  Adjusted n_components to #{n_components} (was #{@n_components}) for #{n_samples} samples"
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
        @logger.warn "Warning: Dimensionality reduction requires ClusterKit. Using original embeddings."
        embeddings
      rescue => e
        @logger.warn "Warning: Dimensionality reduction failed: #{e.message}"
        embeddings
      end
    end
    
    private
    
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
  end
end