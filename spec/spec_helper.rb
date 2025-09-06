# frozen_string_literal: true

require "bundler/setup"
require "topical"

# Helper methods for testing
module TestHelpers
  # Generate simple test embeddings
  def generate_test_embeddings(n_samples: 100, n_features: 50, n_clusters: 3)
    embeddings = []
    
    n_samples.times do |i|
      cluster = i % n_clusters
      # Create embeddings with some cluster structure
      base = Array.new(n_features) { cluster * 2.0 + rand(-0.5..0.5) }
      embeddings << base
    end
    
    embeddings
  end
  
  # Generate test documents
  def generate_test_documents(n_samples: 100)
    topics = [
      ["machine learning", "neural network", "deep learning", "model", "training"],
      ["database", "SQL", "query", "index", "performance"],
      ["web development", "JavaScript", "React", "frontend", "API"]
    ]
    
    n_samples.times.map do |i|
      cluster = i % topics.size
      words = topics[cluster].sample(3).join(" ")
      "This is a document about #{words}. " * 3
    end
  end
end

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = ".rspec_status"

  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end
  
  # Include test helpers
  config.include TestHelpers
  
  # Configure output
  config.formatter = :documentation if ENV['VERBOSE']
  
  # Run specs in random order
  config.order = :random
  Kernel.srand config.seed
end
