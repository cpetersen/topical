# frozen_string_literal: true

require 'clusterkit'

module Topical
  module Clustering
    # Adapter for ClusterKit's K-means implementation
    class KMeansAdapter < Adapter
      def initialize(k: 5, random_seed: nil)
        @k = k
        @random_seed = random_seed
        
        @clusterer = ClusterKit::Clustering::KMeans.new(
          k: k,
          random_seed: random_seed
        )
      end
      
      def fit_predict(embeddings)
        labels = @clusterer.fit_predict(embeddings)
        @n_clusters = @k
        @n_noise_points = 0  # K-means doesn't have noise points
        labels
      end
      
      def fit(embeddings)
        @clusterer.fit(embeddings)
        self
      end
      
      def predict(embeddings)
        @clusterer.predict(embeddings)
      end
      
      # Access cluster centers
      def cluster_centers
        @clusterer.cluster_centers
      end
      
      # Access to underlying ClusterKit object if needed
      attr_reader :clusterer
    end
  end
end