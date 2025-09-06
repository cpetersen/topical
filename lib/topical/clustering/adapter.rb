# frozen_string_literal: true

module Topical
  module Clustering
    # Base adapter class for clustering algorithms
    class Adapter
      def fit_predict(embeddings)
        raise NotImplementedError, "Subclasses must implement fit_predict"
      end
      
      def fit(embeddings)
        raise NotImplementedError, "Subclasses must implement fit"
      end
      
      def predict(embeddings)
        raise NotImplementedError, "Subclasses must implement predict"
      end
      
      # Number of clusters found (excluding noise)
      def n_clusters
        @n_clusters || 0
      end
      
      # Number of noise points (labeled as -1)
      def n_noise_points
        @n_noise_points || 0
      end
    end
  end
end
