# Topical

Topic modeling for Ruby using modern clustering algorithms. Extract meaningful topics from document embeddings using HDBSCAN clustering and c-TF-IDF term extraction.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'topical'

# Optional: for LLM-powered topic labeling
gem 'red-candle'  
```

And then execute:

    $ bundle install

Or install it yourself as:

    $ gem install topical

## Usage

### Simple Interface

```ruby
require 'topical'

# You need embeddings and documents
embeddings = [...] # Array of embedding vectors
documents = [...]  # Array of document strings

# Extract topics
topics = Topical.extract(
  embeddings: embeddings,
  documents: documents
)

# Work with topics
topics.each do |topic|
  puts "#{topic.label}: #{topic.size} documents"
  puts "Terms: #{topic.terms.join(', ')}"
  puts "Sample: #{topic.representative_docs(k: 1).first}"
end
```

### Advanced Usage

```ruby
# Create engine with custom configuration
engine = Topical::Engine.new(
  clustering_method: :hdbscan,    # or :kmeans
  min_cluster_size: 5,             # Minimum documents per topic
  min_samples: 3,                  # HDBSCAN parameter
  reduce_dimensions: true,          # Auto-reduce high-dim embeddings
  n_components: 50,                 # Target dimensions for reduction
  labeling_method: :hybrid,         # :term_based, :llm_based, :hybrid
  verbose: true                     # Show progress
)

# Fit the model
topics = engine.fit(embeddings, documents)

# Transform new documents
new_topics = engine.transform(new_embeddings, new_documents)

# Save and load models
engine.save("model.json")
loaded_engine = Topical::Engine.load("model.json")
```

### Topic Object API

```ruby
topic.id                      # Cluster ID
topic.label                   # Human-readable label
topic.terms                   # Top distinctive terms
topic.documents              # Documents in this topic
topic.size                   # Number of documents
topic.coherence              # Quality score (0-1)
topic.representative_docs(k: 3)  # Most representative documents
```

## How It Works

1. **Clustering**: Groups similar embeddings using HDBSCAN or K-means
2. **Term Extraction**: Finds distinctive terms per topic using c-TF-IDF
3. **Labeling**: Generates human-readable labels (term-based or LLM-powered)
4. **Metrics**: Computes topic quality scores (coherence, distinctiveness)

## Dependencies

- **Required**: `clusterkit` - For HDBSCAN clustering and dimensionality reduction
- **Optional**: `red-candle` - For enhanced LLM-powered topic labeling

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and the created tag, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/cpetersen/topical. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [code of conduct](https://github.com/cpetersen/topical/blob/main/CODE_OF_CONDUCT.md).

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## Code of Conduct

Everyone interacting in the Topical project's codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/cpetersen/topical/blob/main/CODE_OF_CONDUCT.md).