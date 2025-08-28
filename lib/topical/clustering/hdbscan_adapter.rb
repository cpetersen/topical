# frozen_string_literal: true

require 'clusterkit'

module Topical
  module Clustering
    # Adapter for ClusterKit's HDBSCAN implementation
    class HDBSCANAdapter < Adapter
      def initialize(min_cluster_size: 5, min_samples: 3, metric: 'euclidean')
        @min_cluster_size = min_cluster_size
        @min_samples = min_samples
        @metric = metric
        
        @clusterer = ClusterKit::Clustering::HDBSCAN.new(
          min_cluster_size: min_cluster_size,
          min_samples: min_samples,
          metric: metric
        )
      end
      
      def fit_predict(embeddings)
        labels = @clusterer.fit_predict(embeddings)
        update_stats(labels)
        labels
      end
      
      def fit(embeddings)
        @clusterer.fit(embeddings)
        self
      end
      
      def predict(embeddings)
        # HDBSCAN doesn't have a separate predict method
        # For new points, we'd need to use approximate prediction
        if @clusterer.respond_to?(:approximate_predict)
          @clusterer.approximate_predict(embeddings)
        else
          raise NotImplementedError, "HDBSCAN does not support prediction on new data"
        end
      end
      
      # Access to underlying ClusterKit object if needed
      attr_reader :clusterer
      
      private
      
      def update_stats(labels)
        @n_noise_points = labels.count(-1)
        unique_labels = labels.uniq.reject { |l| l == -1 }
        @n_clusters = unique_labels.length
      end
    end
  end
end