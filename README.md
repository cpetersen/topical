<img src="/docs/assets/topical-wide.png" alt="ragnar" height="80px">

Topic modeling for Ruby using modern clustering algorithms. Extract meaningful topics from document embeddings using HDBSCAN clustering and c-TF-IDF term extraction.

## Quick Start (requires red-candle)

```bash
# Install the gem
gem install topical

# Install red-candle so we can generate embeddings
gem install red-candle

# Try it out immediately in IRB
irb
```

```ruby
require 'topical'
require 'red-candle'

# Initialize embedding model
embedder = Candle::EmbeddingModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Your documents
documents = [
  # Finance/Economics Topic
  "The Federal Reserve raised interest rates to combat inflation pressures",
  "Stock markets rallied on positive earnings reports from tech companies",
  "Global supply chain disruptions continue to affect consumer prices",
  "Cryptocurrency markets experienced significant volatility this quarter",
  "Central banks coordinate policy to address economic uncertainty",
  "Corporate bond yields rise as investors seek safer assets",
  "Emerging markets face capital outflows amid dollar strength",

  # Technology/AI Topic
  "New AI breakthrough in natural language processing announced by researchers",
  "Machine learning transforms healthcare diagnostics and treatment planning",
  "Quantum computing reaches new milestone in error correction",
  "Open source community releases major updates to popular frameworks",
  "Cloud computing adoption accelerates across enterprise sectors",
  "Cybersecurity threats evolve with sophisticated ransomware attacks",
  "Artificial intelligence ethics guidelines proposed by tech consortium",

  # Healthcare/Medical Topic
  "Clinical trials show promising results for new cancer immunotherapy",
  "Telemedicine adoption continues to reshape patient care delivery",
  "Gene editing techniques advance treatment for rare diseases",
  "Mental health awareness campaigns gain momentum globally",
  "Vaccine development accelerates using mRNA technology platforms",
  "Healthcare systems invest in digital transformation initiatives",
  "Personalized medicine approaches show improved patient outcomes",

  # Climate/Environment Topic
  "Renewable energy investments surpass fossil fuel spending globally",
  "Climate scientists warn of accelerating Arctic ice melt",
  "Carbon capture technology receives significant government funding",
  "Sustainable agriculture practices reduce environmental impact",
  "Electric vehicle adoption reaches record levels worldwide",
  "Ocean conservation efforts expand marine protected areas",
  "Green hydrogen emerges as key solution for industrial decarbonization",

  # Sports Topic
  "Championship team breaks decades-old winning streak record",
  "Olympic athletes prepare for upcoming international competition",
  "Sports analytics revolutionize player performance evaluation",
  "Major league implements new rules to improve game pace",
  "Youth sports participation increases following pandemic recovery",
  "Stadium technology enhances fan experience with augmented reality",
  "Professional athletes advocate for mental health support"
]

# Generate embeddings
embeddings = documents.map { |doc| embedder.embedding(doc).first.to_a }

# Extract topics with HDBSCAN clustering
engine = Topical::Engine.new(
  clustering_method: :hdbscan,
  min_cluster_size: 5,
  verbose: true
)

topics = engine.fit(embeddings: embeddings, documents: documents)

# Analyze results
topics.each do |topic|
  puts "\nTopic: #{topic.label}"
  puts "Size: #{topic.size} documents"
  puts "Coherence: #{topic.coherence.round(3)}"
  puts "Top terms: #{topic.terms.first(10).join(', ')}"
  puts "\nRepresentative documents:"
  topic.representative_docs(k: 3).each { |doc| puts "  - #{doc[0..100]}..." }
end

# Check for outliers
outliers = engine.outliers
puts "\nOutliers: #{outliers.length} documents"
```

### Advanced Configuration

```ruby
# Create engine with custom configuration
engine = Topical::Engine.new(
  # Clustering options
  clustering_method: :hdbscan,    # :hdbscan or :kmeans
  min_cluster_size: 3,            # Minimum documents per topic (HDBSCAN)
  min_samples: 5,                  # Core points needed (HDBSCAN)
  k: 20,                           # Number of topics (K-means only)

  # Dimensionality reduction
  reduce_dimensions: true,          # Auto-reduce high-dim embeddings with UMAP
  n_components: 50,                 # Target dimensions for reduction

  # Labeling options
  labeling_method: :term_based,     # Fast, reliable term-based labeling

  # Other options
  verbose: true                     # Show progress
)

# Fit the model
topics = engine.fit(embeddings: embeddings, documents: documents)

# Save and load models
engine.save("topic_model.json")
loaded = Topical::Engine.load("topic_model.json")

# Transform: Assign new documents to existing topics
# Note: transform does NOT create new topics - it assigns documents to the closest existing topic
new_documents = [
  # These will be assigned to existing topics based on similarity
  "Stock market reaches all-time high amid economic recovery",  # Should go to Finance
  "New smartphone features AI-powered camera system",           # Should go to Technology
  "Clinical study reveals breakthrough in diabetes treatment",  # Should go to Healthcare
  "Record heat wave highlights climate change urgency"          # Should go to Climate
]
new_embeddings = new_documents.map { |doc| embedder.embedding(doc).first.to_a }

# Returns array of topic IDs that each document was assigned to
assigned_topic_ids = engine.transform(embeddings: new_embeddings, documents: new_documents)

# See which topics the new documents were assigned to
assigned_topic_ids.each_with_index do |topic_id, idx|
  topic = engine.get_topic(topic_id)
  if topic
    puts "Document #{idx}: Assigned to Topic '#{topic.label}'"
    puts "  Document: #{new_documents[idx]}"
  else
    puts "Document #{idx}: Marked as outlier (no matching topic)"
  end
end

# Get specific topic
topic = engine.get_topic(0)
```

### Understanding Transform vs Fit

- **`fit`**: Discovers topics from your training documents. Creates new topic clusters.
- **`transform`**: Assigns new documents to existing topics discovered during `fit`. Does NOT create new topics.

If you have documents that represent a completely new topic not seen during training:
1. They may be assigned to the closest existing topic (even if not very similar)
2. They may be marked as outliers if using HDBSCAN (returned as topic_id -1)
3. To discover new topics, you need to analyze them separately or re-fit

### Detecting New Topics

Yes, you can run `fit` on just the new documents to discover their topics independently! This is useful for:
- Detecting topic drift over time
- Identifying emerging themes
- Validating if new content fits your existing model

See [examples/detect_new_topics.rb](examples/detect_new_topics.rb) for a complete example.

```ruby
# To discover new topics, you have several options:

# Option 1: Fit only on new documents to discover their topics
new_engine = Topical::Engine.new(
  clustering_method: :hdbscan,
  min_cluster_size: 3  # May need to adjust for small batches
)
new_topics = new_engine.fit(embeddings: new_embeddings, documents: new_documents)
puts "Found #{new_topics.size} topics in new documents"

# Option 2: Check if new documents are outliers (potential new topic)
assigned_ids = engine.transform(embeddings: new_embeddings)
outlier_indices = assigned_ids.each_index.select { |i| assigned_ids[i] == -1 }
if outlier_indices.size > 3  # If many outliers, might be new topic
  puts "#{outlier_indices.size} documents don't fit existing topics - potential new topic!"
  outlier_docs = outlier_indices.map { |i| new_documents[i] }
  outlier_embeds = outlier_indices.map { |i| new_embeddings[i] }

  # Cluster just the outliers
  outlier_engine = Topical::Engine.new(min_cluster_size: 3)
  outlier_topics = outlier_engine.fit(embeddings: outlier_embeds, documents: outlier_docs)
end

# Option 3: Incremental topic discovery (combine old + new and re-fit)
all_documents = original_documents + new_documents
all_embeddings = original_embeddings + new_embeddings
updated_topics = engine.fit(embeddings: all_embeddings, documents: all_documents)

# Option 4: Compare similarity scores to detect poor fits
assigned_ids = engine.transform(embeddings: new_embeddings)
similarities = new_embeddings.map.with_index do |embed, idx|
  topic_id = assigned_ids[idx]
  next nil if topic_id == -1

  topic = engine.get_topic(topic_id)
  # Calculate distance to topic centroid (simplified)
  # In practice, you'd compute actual distance to topic center
  { document: new_documents[idx], topic: topic.label, similarity: rand(0.3..1.0) }
end

low_similarity = similarities.compact.select { |s| s[:similarity] < 0.5 }
if low_similarity.size > 3
  puts "#{low_similarity.size} documents have low similarity - might be new topic"
end
```

### Topic Analysis

```ruby
# Access topic properties
topic.id                          # Cluster ID
topic.label                        # Human-readable label
topic.terms                        # Top distinctive terms (c-TF-IDF)
topic.documents                    # All documents in topic
topic.size                         # Number of documents
topic.coherence                    # Topic quality score (0-1)
topic.centroid                     # Topic centroid in embedding space

# Get representative documents
topic.representative_docs(k: 5)    # 5 most representative docs

# Convert to hash for serialization
topic.to_h

# Compute metrics across all topics
diversity = Topical::Metrics.compute_diversity(topics)
coverage = Topical::Metrics.compute_coverage(topics, documents.count + new_documents.count)
```

## Clustering Methods

### HDBSCAN (Hierarchical Density-Based Clustering)
- **Pros**: Automatically determines number of topics, identifies outliers, handles varying densities
- **Cons**: Requires tuning min_cluster_size and min_samples
- **When to use**: When you don't know the number of topics in advance

### K-means
- **Pros**: Fast, deterministic with same seed, always assigns all documents
- **Cons**: Must specify k (number of topics), no outlier detection
- **When to use**: When you know approximately how many topics to expect

## Term Extraction

Topical uses **c-TF-IDF** (class-based TF-IDF) to find distinctive terms for each topic:
- Higher scores for terms frequent in topic but rare in other topics
- Automatically filters stop words
- Configurable minimum/maximum word lengths

## Topic Labeling

Topical uses **term-based labeling** - fast, reliable labels generated from the most distinctive terms in each topic cluster. Labels are created by combining the top 2-3 terms that best characterize each topic.

### Advanced: LLM-Powered Summaries

For richer topic analysis, you can combine Topical's clustering with red-candle's LLM capabilities. See `examples/topic_summaries_with_llm.rb` for a complete example of generating detailed topic summaries using your choice of LLM.

## Dependencies

- **Required**: `clusterkit` - For HDBSCAN clustering and UMAP dimensionality reduction
- **Optional**: `red-candle` - For generating embeddings in examples and advanced LLM summaries

## Performance Tips

1. **Dimensionality Reduction**: For embeddings with >100 dimensions, enable `reduce_dimensions: true`
2. **Batch Processing**: Process documents in batches of 1000-5000 for large datasets
3. **Caching**: Save fitted models with `engine.save()` to avoid recomputation
4. **Parallel Processing**: Generate embeddings in parallel when possible

## Examples

Check out the `examples/` directory for complete examples:
- `quick_demo.rb` - Simple demonstration with mock data
- `topic_summaries_with_llm.rb` - Advanced example showing how to generate detailed topic summaries using red-candle LLM

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt.

To install this gem onto your local machine, run `bundle exec rake install`.

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/scientist-labs/topical.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
