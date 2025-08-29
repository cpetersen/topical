#!/usr/bin/env ruby

require 'bundler/setup'
require 'topical'

puts "🎯 Topical Quick Demo"
puts "=" * 50
puts

# Create some sample documents about different topics
documents = [
  # Ruby/Rails cluster
  "Ruby is a dynamic programming language with elegant syntax",
  "Rails is a web framework written in Ruby for building web applications",
  "Ruby on Rails follows the MVC pattern and convention over configuration",
  
  # Python/ML cluster
  "Python is great for machine learning and data science applications",
  "TensorFlow and PyTorch are popular machine learning frameworks in Python",
  "Machine learning models need training data and validation sets",
  "Deep learning uses neural networks with multiple layers",
  
  # JavaScript/Web cluster
  "JavaScript runs in browsers and Node.js for full-stack development",
  "React and Vue are modern JavaScript frameworks for building UIs",
  "Web development involves HTML, CSS, and JavaScript",
  "Frontend frameworks help build interactive user interfaces",
  
  # Database cluster
  "SQL databases use structured queries to manage relational data",
  "NoSQL databases like MongoDB store documents in flexible schemas",
  "Database indexing improves query performance significantly"
]

puts "📚 Processing #{documents.length} documents..."
puts

# Create simple mock embeddings based on keywords
# In real usage, you'd use actual embeddings from red-candle or similar
embeddings = documents.map do |doc|
  text = doc.downcase
  [
    # Feature engineering based on topic keywords
    text.include?("ruby") || text.include?("rails") ? 1.0 : 0.0,
    text.include?("python") || text.include?("machine") || text.include?("learning") || text.include?("neural") ? 1.0 : 0.0,
    text.include?("javascript") || text.include?("react") || text.include?("vue") || text.include?("web") || text.include?("frontend") ? 1.0 : 0.0,
    text.include?("database") || text.include?("sql") || text.include?("mongodb") || text.include?("query") ? 1.0 : 0.0,
    rand(-0.1..0.1),  # Add small random noise
    rand(-0.1..0.1)
  ]
end

# Extract topics using K-means (since we know we have ~4 topic areas)
puts "🔍 Extracting topics with K-means clustering..."
topics = Topical.extract(
  embeddings: embeddings,
  documents: documents,
  clustering_method: :kmeans,
  k: 4,
  verbose: false
)

puts "✅ Found #{topics.length} topics!"
puts

# Display the results
topics.each_with_index do |topic, i|
  puts "━" * 50
  puts "📌 Topic #{i + 1}: #{topic.label}"
  puts "━" * 50
  puts "📊 Size: #{topic.size} documents"
  puts "🔤 Key terms: #{topic.terms.first(6).join(', ')}"
  puts "📈 Coherence: #{(topic.coherence * 100).round(1)}%"
  puts
  puts "📄 Documents in this topic:"
  topic.documents.each_with_index do |doc, j|
    preview = doc.length > 60 ? "#{doc[0..60]}..." : doc
    puts "   #{j + 1}. #{preview}"
  end
  puts
end

# Show topic diversity
diversity = Topical::Metrics.compute_diversity(topics)
puts "━" * 50
puts "📊 Overall topic diversity: #{(diversity * 100).round(1)}%"
puts "💡 Higher diversity means topics are more distinct from each other"
puts

# Test HDBSCAN clustering (density-based, finds optimal number of clusters)
puts "━" * 50
puts "🔍 Now trying HDBSCAN clustering (automatic topic detection)..."
puts

engine = Topical::Engine.new(
  clustering_method: :hdbscan,
  min_cluster_size: 3,  # Minimum 3 docs per topic
  min_samples: 2
)

hdbscan_topics = engine.fit(embeddings, documents)
outliers = engine.outliers

puts "✅ HDBSCAN found #{hdbscan_topics.length} topics"
puts "🔸 Outliers: #{outliers.length} documents"

if outliers.any?
  puts "\nDocuments marked as outliers (don't fit well in any topic):"
  outliers.each { |doc| puts "  - #{doc[0..60]}..." }
end

puts
puts "━" * 50
puts "🎉 Demo complete! Try it with your own documents and real embeddings."
puts
puts "💡 Tip: Install red-candle to generate real embeddings:"
puts "   gem install red-candle"
puts "   Then use: RedCandle::Embedding.new('model-name').embed(text)"