# frozen_string_literal: true

require 'spec_helper'

RSpec.describe "Term extraction verification" do
  it "actually extracts meaningful terms from real topics" do
    # Create clear, distinct documents
    tech_docs = [
      "Python programming and machine learning with neural networks",
      "JavaScript React frontend development with modern web APIs",
      "Database SQL queries and index optimization for performance",
      "Software engineering practices and agile development methods",
      "Cloud computing AWS infrastructure and DevOps automation"
    ]
    
    finance_docs = [
      "Stock market trading and investment portfolio management",
      "Financial analysis quarterly earnings and profit margins",
      "Banking regulations compliance and risk assessment strategies",
      "Corporate bonds equity markets and hedge fund investments",
      "Economic indicators inflation rates and monetary policy"
    ]
    
    science_docs = [
      "Scientific research methodology and experimental design",
      "Chemistry laboratory analysis and molecular structures",
      "Physics quantum mechanics and particle acceleration",
      "Biology genetics DNA sequencing and cell division",
      "Astronomy telescope observations and planetary motion"
    ]
    
    all_docs = tech_docs + finance_docs + science_docs
    
    # Create simple embeddings that cluster properly
    embeddings = []
    # Tech cluster
    5.times { embeddings << [1.0, 0.0, 0.0] }
    # Finance cluster  
    5.times { embeddings << [0.0, 1.0, 0.0] }
    # Science cluster
    5.times { embeddings << [0.0, 0.0, 1.0] }
    
    engine = Topical::Engine.new(
      clustering_method: :kmeans,
      k: 3,
      verbose: true
    )
    
    puts "\n=== Running topic extraction with verbose output ==="
    topics = engine.fit(embeddings, all_docs)
    
    puts "\n=== Topics Found ==="
    topics.each do |topic|
      puts "\nTopic #{topic.id}: '#{topic.label}'"
      puts "  Size: #{topic.size} documents"
      puts "  Terms: #{topic.terms.first(10).join(', ')}"
      puts "  Sample doc: #{topic.documents.first[0..100]}..."
      puts "  Coherence: #{topic.coherence.round(3)}"
    end
    
    # Verify we got sensible terms
    all_terms = topics.flat_map(&:terms)
    
    # Should include domain-specific terms
    expect(all_terms).to include(match(/programming|javascript|database|software/i))
    expect(all_terms).to include(match(/market|investment|financial|banking/i))
    expect(all_terms).to include(match(/research|chemistry|physics|biology/i))
    
    # Should NOT include stop words
    stop_words = %w[the and with for]
    stop_words.each do |word|
      expect(all_terms).not_to include(word)
    end
    
    # Each topic should have different top terms
    top_terms_per_topic = topics.map { |t| t.terms.first(5) }
    top_terms_per_topic.combination(2).each do |terms1, terms2|
      # Should have minimal overlap
      overlap = (terms1 & terms2).size
      expect(overlap).to be <= 1  # Allow at most 1 common term
    end
  end
  
  describe "c-TF-IDF algorithm" do
    it "gives higher scores to terms unique to a topic" do
      extractor = Topical::Extractors::TermExtractor.new
      
      topic_docs = [
        "Ruby programming with Rails framework",
        "Ruby gems and bundler for dependency management"
      ]
      
      all_docs = topic_docs + [
        "Python programming with Django framework",
        "JavaScript programming with React framework"
      ]
      
      terms = extractor.extract_distinctive_terms(
        topic_docs: topic_docs,
        all_docs: all_docs,
        top_n: 5
      )
      
      # "Ruby" should rank high because it's distinctive to this topic
      # "programming" should rank lower because it appears in all topics
      expect(terms.first(2)).to include("ruby")
      expect(terms.first(2)).not_to include("programming")
    end
  end
end