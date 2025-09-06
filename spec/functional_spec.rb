# frozen_string_literal: true

require 'spec_helper'

RSpec.describe "Functional topic modeling" do
  describe "with real embeddings and documents" do
    let(:embeddings) do
      # Create 3 distinct clusters in embedding space
      embeddings = []
      
      # Cluster 1: Tech/Programming (centered at ~[1, 0, 0, ...])
      20.times do
        base = [1.0 + rand(-0.1..0.1)]
        base += Array.new(9) { rand(-0.1..0.1) }
        embeddings << base
      end
      
      # Cluster 2: Business/Finance (centered at ~[0, 1, 0, ...])
      20.times do
        base = [rand(-0.1..0.1), 1.0 + rand(-0.1..0.1)]
        base += Array.new(8) { rand(-0.1..0.1) }
        embeddings << base
      end
      
      # Cluster 3: Science/Research (centered at ~[0, 0, 1, ...])
      20.times do
        base = [rand(-0.1..0.1), rand(-0.1..0.1), 1.0 + rand(-0.1..0.1)]
        base += Array.new(7) { rand(-0.1..0.1) }
        embeddings << base
      end
      
      # Add some noise/outliers
      5.times do
        embeddings << Array.new(10) { rand(-0.5..0.5) }
      end
      
      embeddings
    end
    
    let(:documents) do
      docs = []
      
      # Tech documents
      20.times do |i|
        docs << "Python programming with neural networks and deep learning frameworks. " \
                "Machine learning models require training data and optimization algorithms. " \
                "Software engineering practices for artificial intelligence systems."
      end
      
      # Business documents  
      20.times do |i|
        docs << "Financial markets show strong quarterly earnings and profit margins. " \
                "Investment portfolio management strategies for corporate bonds and equity. " \
                "Business analytics and revenue growth through market expansion."
      end
      
      # Science documents
      20.times do |i|
        docs << "Scientific research methodology and experimental design principles. " \
                "Laboratory analysis of chemical compounds and molecular structures. " \
                "Peer-reviewed journals publish groundbreaking discoveries in physics."
      end
      
      # Noise documents
      5.times do |i|
        docs << "Random text that doesn't fit any particular category. " \
                "Miscellaneous content with various unrelated topics mixed together. " \
                "General information without specific domain focus."
      end
      
      docs
    end
    
    describe "HDBSCAN clustering" do
      it "finds distinct topics in the data" do
        engine = Topical::Engine.new(
          clustering_method: :hdbscan,
          min_cluster_size: 5,
          min_samples: 2,
          reduce_dimensions: false,  # Our test embeddings are already low-dim
          labeling_method: :term_based,
          verbose: false
        )
        
        topics = engine.fit(embeddings: embeddings, documents: documents)
        
        # Should find at least 2 topics (possibly 3, depending on clustering)
        expect(topics).to be_an(Array)
        expect(topics.length).to be >= 2
        
        # Each topic should have documents
        topics.each do |topic|
          expect(topic.documents.length).to be >= 5  # min_cluster_size
        end
      end
      
      it "extracts distinctive terms for each topic" do
        engine = Topical::Engine.new(
          clustering_method: :hdbscan,
          min_cluster_size: 5
        )
        
        topics = engine.fit(embeddings: embeddings, documents: documents)
        
        topics.each do |topic|
          # Should have extracted terms
          expect(topic.terms).to be_an(Array)
          expect(topic.terms).not_to be_empty
          expect(topic.terms.first).to be_a(String)
          
          # Terms should be meaningful words (not stopwords)
          expect(topic.terms.first.length).to be > 3
        end
      end
      
      it "generates labels for topics" do
        engine = Topical::Engine.new(
          clustering_method: :hdbscan,
          min_cluster_size: 5,
          labeling_method: :term_based
        )
        
        topics = engine.fit(embeddings: embeddings, documents: documents)
        
        topics.each do |topic|
          expect(topic.label).not_to be_nil
          expect(topic.label).to be_a(String)
          expect(topic.label.length).to be > 0
          
          # Should not be just "Topic N"
          expect(topic.label).not_to match(/^Topic \d+$/)
        end
      end
      
      it "identifies noise points" do
        engine = Topical::Engine.new(
          clustering_method: :hdbscan,
          min_cluster_size: 25,  # Very high threshold - clusters need 25+ docs
          min_samples: 5,        # Higher min_samples too
          reduce_dimensions: false  # Use original embeddings for predictable results
        )
        
        topics = engine.fit(embeddings: embeddings, documents: documents)
        
        # With clusters of 20 docs and min_cluster_size of 25, all should be noise
        expect(engine.clustering_adapter.n_noise_points).to be > 40
      end
    end
    
    describe "K-means clustering" do
      it "creates exactly k topics" do
        engine = Topical::Engine.new(
          clustering_method: :kmeans,
          k: 3,
          reduce_dimensions: false
        )
        
        topics = engine.fit(embeddings: embeddings, documents: documents)
        
        expect(topics.length).to eq(3)
        
        # All documents should be assigned
        total_docs = topics.sum(&:size)
        expect(total_docs).to eq(documents.length)
      end
      
      it "computes topic centroids" do
        engine = Topical::Engine.new(
          clustering_method: :kmeans,
          k: 3
        )
        
        topics = engine.fit(embeddings: embeddings, documents: documents)
        
        topics.each do |topic|
          centroid = topic.centroid
          expect(centroid).to be_an(Array)
          expect(centroid.length).to eq(embeddings.first.length)
          
          # Centroid values should be reasonable
          centroid.each do |value|
            expect(value).to be_a(Float)
            expect(value.abs).to be < 10  # Reasonable bounds
          end
        end
      end
    end
    
    describe "Term extraction" do
      it "extracts different terms for different topics" do
        engine = Topical::Engine.new(
          clustering_method: :kmeans,
          k: 3
        )
        
        topics = engine.fit(embeddings: embeddings, documents: documents)
        
        # Get terms from different topics
        terms_sets = topics.map { |t| Set.new(t.terms[0..5]) }
        
        # Terms should be somewhat distinct between topics
        terms_sets.combination(2).each do |set1, set2|
          overlap = (set1 & set2).size
          # Some overlap is OK, but not complete overlap
          expect(overlap).to be < [set1.size, set2.size].min
        end
      end
      
      it "filters out stop words" do
        engine = Topical::Engine.new(clustering_method: :kmeans, k: 3)
        topics = engine.fit(embeddings: embeddings, documents: documents)
        
        stop_words = %w[the be to of and a in that have with for]
        
        topics.each do |topic|
          topic.terms.each do |term|
            expect(stop_words).not_to include(term.downcase)
          end
        end
      end
    end
    
    describe "Representative documents" do
      it "finds documents closest to topic centroid" do
        engine = Topical::Engine.new(clustering_method: :kmeans, k: 3)
        topics = engine.fit(embeddings: embeddings, documents: documents)
        
        topics.each do |topic|
          rep_docs = topic.representative_docs(k: 3)
          
          expect(rep_docs).to be_an(Array)
          expect(rep_docs.length).to be <= 3
          expect(rep_docs.length).to be <= topic.size
          
          rep_docs.each do |doc|
            expect(doc).to be_a(String)
            expect(topic.documents).to include(doc)
          end
        end
      end
    end
    
    describe "Topic quality metrics" do
      it "computes coherence scores" do
        engine = Topical::Engine.new(clustering_method: :kmeans, k: 3)
        topics = engine.fit(embeddings: embeddings, documents: documents)
        
        topics.each do |topic|
          expect(topic.coherence).to be_a(Float)
          expect(topic.coherence).to be_between(0, 1).inclusive
        end
      end
    end
    
    describe "Edge cases" do
      it "handles single cluster data" do
        # All same embeddings
        same_embeddings = Array.new(20) { [1.0] * 10 }
        same_docs = Array.new(20) { "Same document repeated" }
        
        engine = Topical::Engine.new(
          clustering_method: :hdbscan,
          min_cluster_size: 5
        )
        
        expect { engine.fit(embeddings: same_embeddings, documents: same_docs) }.not_to raise_error
      end
      
      it "handles very small datasets" do
        small_embeddings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        small_docs = ["doc1", "doc2", "doc3"]
        
        engine = Topical::Engine.new(
          clustering_method: :hdbscan,
          min_cluster_size: 2,
          min_samples: 1
        )
        
        expect { engine.fit(embeddings: small_embeddings, documents: small_docs) }.not_to raise_error
      end
    end
  end
  
  describe "Simple interface" do
    it "works with minimal configuration" do
      embeddings = generate_test_embeddings(n_samples: 50, n_features: 10, n_clusters: 2)
      documents = generate_test_documents(n_samples: 50)
      
      topics = Topical.extract(embeddings: embeddings, documents: documents)
      
      expect(topics).to be_an(Array)
      expect(topics).not_to be_empty
    end
  end
end
