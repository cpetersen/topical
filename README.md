# Topical

Topic modeling for Ruby using modern clustering algorithms. Extract meaningful topics from document embeddings using HDBSCAN clustering and c-TF-IDF term extraction.

## Quick Start

```bash
# Install the gem
gem install topical

# Try it out immediately in IRB
irb
```

```ruby
require 'topical'

# Create some sample documents
documents = [
  "Ruby is a dynamic programming language with elegant syntax",
  "Rails is a web framework written in Ruby for building web applications",
  "Python is great for machine learning and data science applications",
  "TensorFlow and PyTorch are popular machine learning frameworks in Python",
  "JavaScript runs in browsers and Node.js for full-stack development",
  "React and Vue are modern JavaScript frameworks for building UIs",
  "Machine learning models need training data and validation sets",
  "Deep learning uses neural networks with multiple layers",
  "Web development involves HTML, CSS, and JavaScript",
  "Backend development often uses databases and APIs"
]

# Create simple mock embeddings (in practice, use real embeddings from red-candle or other embedding models)
# Here we create 3 distinct clusters based on keywords
embeddings = documents.map do |doc|
  text = doc.downcase
  [
    text.include?("ruby") || text.include?("rails") ? 1.0 : 0.0,  # Ruby cluster
    text.include?("python") || text.include?("machine") || text.include?("learning") ? 1.0 : 0.0,  # ML cluster  
    text.include?("javascript") || text.include?("web") || text.include?("css") ? 1.0 : 0.0,  # Web cluster
    rand(-0.1..0.1)  # Small random noise
  ]
end

# Extract topics
topics = Topical.extract(
  embeddings: embeddings,
  documents: documents,
  clustering_method: :kmeans,
  k: 3
)

# Display results
topics.each do |topic|
  puts "\n📌 #{topic.label}"
  puts "   Documents: #{topic.size}"
  puts "   Key terms: #{topic.terms.first(5).join(', ')}"
  puts "   Sample: \"#{topic.documents.first[0..80]}...\""
end
```

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'topical'

# Optional but recommended: for generating real embeddings
gem 'red-candle'  
```

And then execute:

    $ bundle install

Or install it yourself as:

    $ gem install topical

## Real-World Usage with Embeddings

### Using with red-candle (recommended)

```ruby
require 'topical'
require 'red-candle'

# Initialize embedding model
embedder = RedCandle::Embedding.new("sentence-transformers/all-MiniLM-L6-v2")

# Your documents
documents = [
  "The Federal Reserve raised interest rates to combat inflation",
  "Stock markets rallied on positive earnings reports",
  "New AI breakthrough in natural language processing",
  "Machine learning transforms healthcare diagnostics",
  # ... more documents
]

# Generate embeddings
embeddings = documents.map { |doc| embedder.embed(doc) }

# Extract topics with HDBSCAN clustering
engine = Topical::Engine.new(
  clustering_method: :hdbscan,
  min_cluster_size: 5,
  verbose: true
)

topics = engine.fit(embeddings, documents)

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
  min_cluster_size: 10,            # Minimum documents per topic (HDBSCAN)
  min_samples: 5,                  # Core points needed (HDBSCAN)
  k: 20,                           # Number of topics (K-means only)
  
  # Dimensionality reduction
  reduce_dimensions: true,          # Auto-reduce high-dim embeddings with UMAP
  n_components: 50,                 # Target dimensions for reduction
  
  # Labeling options
  labeling_method: :hybrid,         # :term_based, :llm_based, or :hybrid
  llm_provider: nil,               # Optional: custom LLM provider
  
  # Other options
  verbose: true                     # Show progress
)

# Fit the model
topics = engine.fit(embeddings, documents, metadata: metadata)

# Save and load models
engine.save("topic_model.json")
loaded = Topical::Engine.load("topic_model.json")

# Transform new documents
new_topics = engine.transform(new_embeddings)

# Get specific topic
topic = engine.get_topic(0)
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
coverage = Topical::Metrics.compute_coverage(topics, total_docs)
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

## Topic Labeling Methods

1. **Term-based** (`:term_based`)
   - Fast, uses top distinctive terms
   - No external dependencies
   
2. **LLM-based** (`:llm_based`)
   - High quality, contextual labels
   - Requires red-candle or API provider
   
3. **Hybrid** (`:hybrid`)
   - Best of both: fast with LLM enhancement
   - Falls back to term-based if LLM unavailable

## Dependencies

- **Required**: `clusterkit` - For HDBSCAN clustering and UMAP dimensionality reduction
- **Optional**: `red-candle` - For generating embeddings and LLM-powered topic labeling

## Performance Tips

1. **Dimensionality Reduction**: For embeddings with >100 dimensions, enable `reduce_dimensions: true`
2. **Batch Processing**: Process documents in batches of 1000-5000 for large datasets
3. **Caching**: Save fitted models with `engine.save()` to avoid recomputation
4. **Parallel Processing**: Generate embeddings in parallel when possible

## Examples

Check out the `examples/` directory for complete examples:
- `quick_demo.rb` - Simple demonstration with mock data
- `news_clustering.rb` - Clustering news articles
- `customer_feedback.rb` - Analyzing customer feedback topics
- `research_papers.rb` - Organizing research papers by topic

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt.

To install this gem onto your local machine, run `bundle exec rake install`.

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/cpetersen/topical.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).