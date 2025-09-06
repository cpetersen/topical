#!/usr/bin/env ruby

require 'bundler/setup'
require 'topical'

puts "=== Topical Migration Verification ==="
puts "Version: #{Topical::VERSION}"
puts "LLM Available: #{Topical.llm_available?}"
puts

# Create test data
embeddings = []
documents = []

# Tech cluster
10.times do |i|
  embeddings << [1.0 + rand(-0.1..0.1), rand(-0.1..0.1), rand(-0.1..0.1)]
  documents << "Python programming #{i} with machine learning and neural networks"
end

# Business cluster
10.times do |i|
  embeddings << [rand(-0.1..0.1), 1.0 + rand(-0.1..0.1), rand(-0.1..0.1)]
  documents << "Financial markets #{i} with investment portfolio and trading strategies"
end

# Science cluster
10.times do |i|
  embeddings << [rand(-0.1..0.1), rand(-0.1..0.1), 1.0 + rand(-0.1..0.1)]
  documents << "Scientific research #{i} with experimental methodology and peer review"
end

puts "Testing different clustering methods:"
puts

# Test HDBSCAN
puts "1. HDBSCAN Clustering:"
engine = Topical::Engine.new(
  clustering_method: :hdbscan,
  min_cluster_size: 5,
  verbose: true
)
topics = engine.fit(embeddings, documents)
puts "  Found #{topics.length} topics"
topics.each do |topic|
  puts "    Topic #{topic.id}: #{topic.label} (#{topic.size} docs)"
end
puts

# Test K-means
puts "2. K-means Clustering:"
engine = Topical::Engine.new(
  clustering_method: :kmeans,
  k: 3,
  verbose: false
)
topics = engine.fit(embeddings, documents)
puts "  Found #{topics.length} topics"
topics.each do |topic|
  puts "    Topic #{topic.id}: #{topic.label} (#{topic.size} docs)"
end
puts

# Test convenience method
puts "3. Convenience API:"
topics = Topical.extract(
  embeddings: embeddings,
  documents: documents,
  clustering_method: :kmeans,
  k: 3
)
puts "  Found #{topics.length} topics"
puts

# Test advanced features
puts "4. Advanced Features:"
topic = topics.first
puts "  Representative docs: #{topic.representative_docs(k: 2).length}"
puts "  Coherence: #{topic.coherence.round(3)}"
puts "  Centroid dimensions: #{topic.centroid.length}"
puts

# Test metrics
puts "5. Metrics Module:"
diversity = Topical::Metrics.compute_diversity(topics)
coverage = Topical::Metrics.compute_coverage(topics, documents.length)
puts "  Topic diversity: #{diversity.round(3)}"
puts "  Document coverage: #{coverage.round(3)}"
puts

# Test persistence
puts "6. Model Persistence:"
model_path = "/tmp/topical_test_model.json"
engine.save(model_path)
puts "  Model saved to #{model_path}"

loaded_engine = Topical::Engine.load(model_path)
puts "  Model loaded successfully"
puts "  Loaded topics: #{loaded_engine.topics.length}"
puts

# Test outlier detection
puts "7. Outlier Detection:"
outliers = engine.outliers
puts "  Outliers: #{outliers.length}"
puts

puts "=== All tests passed! Migration successful. ===" 
