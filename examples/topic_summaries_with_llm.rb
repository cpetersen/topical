#!/usr/bin/env ruby
# Advanced example: Using Topical for clustering + red-candle for topic summaries

require 'bundler/setup'
require 'topical'
require 'red-candle'

puts "=== Advanced Topic Summaries Example ==="
puts "Combining Topical clustering with red-candle LLM summarization"
puts

# Sample documents with clear topic clusters
documents = [
  # Finance/Economics
  "The Federal Reserve raised interest rates to combat inflation pressures",
  "Stock markets rallied on positive earnings reports from tech companies",
  "Cryptocurrency markets experienced significant volatility this quarter",
  "Central banks coordinate policy to address economic uncertainty",
  "Corporate bond yields rise as investors seek safer assets",
  
  # Technology/AI  
  "New AI breakthrough in natural language processing announced by researchers",
  "Machine learning transforms healthcare diagnostics and treatment planning",
  "Cloud computing adoption accelerates across enterprise sectors",
  "Cybersecurity threats evolve with sophisticated ransomware attacks",
  "Quantum computing reaches new milestone in error correction",
  
  # Healthcare/Medical
  "Clinical trials show promising results for new cancer immunotherapy",
  "Telemedicine adoption continues to reshape patient care delivery", 
  "Gene editing techniques advance treatment for rare diseases",
  "Mental health awareness campaigns gain momentum globally",
  "Personalized medicine approaches show improved patient outcomes",
  
  # Climate/Environment
  "Renewable energy investments surpass fossil fuel spending globally",
  "Climate scientists warn of accelerating Arctic ice melt",
  "Carbon capture technology receives significant government funding",
  "Electric vehicle adoption reaches record levels worldwide",
  "Sustainable agriculture practices reduce environmental impact"
]

# Step 1: Generate embeddings using red-candle
puts "1. Generating embeddings with red-candle..."
embedder = Candle::EmbeddingModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embeddings = documents.map { |doc| embedder.embedding(doc).first.to_a }

# Step 2: Extract topics using Topical (term-based labeling only)
puts "2. Extracting topics with Topical..."
engine = Topical::Engine.new(
  clustering_method: :hdbscan,
  min_cluster_size: 4,
  labeling_method: :term_based,
  verbose: true
)

topics = engine.fit(embeddings: embeddings, documents: documents)

# Step 3: Generate summaries using red-candle LLM
puts "\n3. Generating topic summaries with LLM..."

# Initialize LLM for summarization
llm = Candle::LLM.from_pretrained(
  "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  gguf_file: "tinyllama-1.1b-chat-v1.0.q4_k_m.gguf"
)

def summarize_topic(topic, llm)
  # Get representative documents for context
  sample_docs = topic.representative_docs(k: 3)
  
  # Simple, clear prompt for summarization
  prompt = <<~PROMPT
    Summarize what connects these documents in 1-2 sentences:
    
    Key terms: #{topic.terms.first(5).join(', ')}
    
    Documents:
    #{sample_docs.map.with_index { |doc, i| "#{i+1}. #{doc}" }.join("\n")}
    
    Summary:
  PROMPT
  
  begin
    summary = llm.generate(prompt).strip
    # Clean up common artifacts
    summary = summary.lines.first&.strip || "Related documents"
    summary = summary.gsub(/^(Summary:|Topic:|Documents:)/i, '').strip
    summary.empty? ? "Documents about #{topic.terms.first(2).join(' and ')}" : summary
  rescue => e
    "Documents about #{topic.terms.first(2).join(' and ')}"
  end
end

# Step 4: Display results with summaries
puts "\n=== Topics with LLM Summaries ==="

topics.each_with_index do |topic, i|
  puts "\n#{i + 1}. Topic: #{topic.label}"
  
  # Generate summary using LLM
  summary = summarize_topic(topic, llm)
  puts "   Summary: #{summary}"
  
  puts "   Size: #{topic.size} documents"
  puts "   Key terms: #{topic.terms.first(8).join(', ')}"
  puts "   Coherence: #{topic.coherence.round(3)}"
  puts "   Sample documents:"
  topic.representative_docs(k: 2).each do |doc|
    puts "   • #{doc[0..80]}..."
  end
end

# Step 5: Show outliers
outliers = engine.outliers
if outliers.any?
  puts "\nOutliers (#{outliers.length} documents):"
  outliers.each { |doc| puts "  • #{doc[0..60]}..." }
end

puts "\n=== Key Benefits of This Approach ==="
puts "• Topical handles clustering expertly (fast, reliable)"
puts "• Your application controls LLM integration completely"
puts "• Domain-specific prompts for better summaries"
puts "• Easy to swap LLM providers or models"
puts "• Clean separation of concerns"

puts "\nDone! 🎯"