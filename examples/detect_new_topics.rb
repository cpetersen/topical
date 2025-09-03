#!/usr/bin/env ruby
# Example: Detecting emergence of new topics in document streams

require 'topical'
require 'candle'
require 'json'

puts "Loading embedding model..."
embedder = Candle::EmbeddingModel.from_pretrained("jinaai/jina-embeddings-v2-base-en")  # Default ragnar model
puts "Model loaded!"

# Initial documents (3 clear topics)
initial_documents = [
  # Finance
  "Stock market reaches record highs amid economic recovery",
  "Federal Reserve considers interest rate adjustments",
  "Cryptocurrency adoption grows among institutional investors",
  "Banking sector reports strong quarterly earnings",
  "Global trade agreements impact currency markets",
  
  # Technology
  "Artificial intelligence breakthrough in natural language processing",
  "Cloud computing services expand globally",
  "Quantum computing achieves new milestone",
  "Cybersecurity threats evolve with new techniques",
  "Open source community releases major updates",
  
  # Healthcare
  "Clinical trials show promise for new cancer treatment",
  "Telemedicine adoption continues post-pandemic growth",
  "Gene therapy advances for rare diseases",
  "Mental health awareness campaigns expand",
  "Vaccine development using mRNA technology"
]

puts "=" * 60
puts "INITIAL TOPIC MODELING"
puts "=" * 60

# Train initial model
puts "\nGenerating embeddings for #{initial_documents.size} documents..."
initial_embeddings = initial_documents.map.with_index do |doc, i|
  print "." if i % 5 == 0
  embedder.embedding(doc).first.to_a
end
puts " done!"

puts "Creating topic model..."
engine = Topical::Engine.new(
  clustering_method: :kmeans,  # Use k-means for small dataset
  k: 3,  # We know we have 3 distinct topics
  reduce_dimensions: false,  # Don't reduce dimensions for small dataset
  verbose: true     # Show progress
)

puts "Fitting model..."
initial_topics = engine.fit(embeddings: initial_embeddings, documents: initial_documents)
puts "\nFound #{initial_topics.size} initial topics:"
initial_topics.each do |topic|
  puts "  Topic #{topic.id}: #{topic.terms.take(5).join(', ')}"
  puts "    Size: #{topic.size} documents"
end

# New documents arrive - including a new topic (Education)
new_documents = [
  # More finance
  "Market volatility increases ahead of earnings season",
  
  # More tech
  "Machine learning models improve prediction accuracy",
  
  # NEW TOPIC: Education (not in original training)
  "Online learning platforms transform education delivery",
  "Universities adopt hybrid teaching models globally",
  "STEM education initiatives target underserved communities",
  "Educational technology startups receive record funding",
  "Student debt crisis prompts policy discussions",
  "Coding bootcamps address skills gap in workforce"
]

puts "\nGenerating embeddings for new documents..."
new_embeddings = new_documents.map.with_index do |doc, i|
  print "."
  embedder.embedding(doc).first.to_a
end
puts " done!"

puts "\n" + "=" * 60
puts "DETECTING NEW TOPICS IN INCOMING DOCUMENTS"
puts "=" * 60

# Method 1: Try to assign to existing topics
puts "\n1. Checking fit with existing topics..."
assigned_ids = engine.transform(embeddings: new_embeddings)

assigned_ids.each_with_index do |topic_id, idx|
  doc_preview = new_documents[idx][0..50] + "..."
  if topic_id == -1
    puts "  ❌ Outlier: #{doc_preview}"
  else
    topic = engine.get_topic(topic_id)
    puts "  ✓ Topic #{topic_id}: #{doc_preview}"
  end
end

outlier_count = assigned_ids.count(-1)
puts "\nFound #{outlier_count} outliers (documents that don't fit existing topics)"

# Method 2: Cluster new documents independently
puts "\n2. Clustering new documents independently..."
if new_documents.size >= 5  # Need minimum documents for clustering
  new_engine = Topical::Engine.new(
    clustering_method: :kmeans,  # Use kmeans for small datasets
    k: 2,  # Expect ~2 topics in new docs
    verbose: false
  )

  begin
    new_topics_only = new_engine.fit(embeddings: new_embeddings, documents: new_documents)
    puts "Found #{new_topics_only.size} topics in new documents:"
    
    new_topics_only.each do |topic|
      puts "  New Topic: #{topic.terms.take(5).join(', ')}"
      puts "    Documents: #{topic.size}"
      puts "    Sample: #{topic.documents.first[0..60]}..." if topic.documents.any?
    end
  rescue => e
    puts "Could not cluster new documents alone: #{e.message}"
  end
else
  puts "Too few new documents (#{new_documents.size}) for independent clustering"
end

# Method 3: Identify potential new topic from outliers
if outlier_count >= 3
  puts "\n3. Analyzing outliers for potential new topic..."
  outlier_indices = assigned_ids.each_index.select { |i| assigned_ids[i] == -1 }
  outlier_docs = outlier_indices.map { |i| new_documents[i] }
  outlier_embeds = outlier_indices.map { |i| new_embeddings[i] }
  
  # Check if outliers are similar to each other (potential new topic)
  puts "Outlier documents:"
  outlier_docs.each { |doc| puts "  - #{doc[0..60]}..." }
  
  # Try clustering just the outliers
  if outlier_docs.size >= 3
    # Check similarity among outliers to see if they form coherent group
    puts "\n🎯 Multiple outliers detected - potential new topic!"
    puts "Consider re-training model to discover new topics"
  end
end

# Method 4: Re-fit with all documents to discover new structure
puts "\n4. Re-fitting with all documents..."
all_docs = initial_documents + new_documents
all_embeds = initial_embeddings + new_embeddings

updated_engine = Topical::Engine.new(
  clustering_method: :kmeans,
  k: 4,  # Try 4 topics since we might have a new one
  reduce_dimensions: false,  # Don't reduce dimensions for small dataset
  verbose: false
)

updated_topics = updated_engine.fit(embeddings: all_embeds, documents: all_docs)
puts "After re-fitting with all documents: #{updated_topics.size} topics"

if updated_topics.size > initial_topics.size
  puts "✨ NEW TOPICS EMERGED! (#{updated_topics.size - initial_topics.size} new)"
  updated_topics.each do |topic|
    puts "  Topic: #{topic.terms.take(5).join(', ')} (#{topic.size} docs)"
  end
else
  puts "No new topics detected after re-fitting"
end

puts "\n" + "=" * 60
puts "SUMMARY"
puts "=" * 60
puts "Initial topics: #{initial_topics.size}"
puts "Outliers in new batch: #{outlier_count}/#{new_documents.size}"
puts "Topics after re-fit: #{updated_topics.size}"
puts "\nRecommendation:"
if outlier_count > new_documents.size * 0.3
  puts "⚠️  High outlier rate suggests emerging new topic(s)"
  puts "Consider re-training your topic model with recent documents"
else
  puts "✓ New documents mostly fit existing topics"
  puts "Current model appears adequate"
end