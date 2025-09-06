# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Topical::ModelSerializer do
  let(:simple_embeddings) { [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]] }
  let(:simple_documents) { ["doc1", "doc2", "doc3", "doc4"] }
  let(:engine) do
    engine = Topical::Engine.new(
      clustering_method: :kmeans,
      k: 2,
      reduce_dimensions: false,
      labeling_method: :term_based
    )
    engine.fit(embeddings: simple_embeddings, documents: simple_documents)
    engine
  end

  describe ".save" do
    it "saves engine to JSON file" do
      temp_file = "/tmp/test_serializer_#{Time.now.to_i}.json"
      
      begin
        Topical::ModelSerializer.save(engine, temp_file)
        
        expect(File.exist?(temp_file)).to be true
        content = JSON.parse(File.read(temp_file), symbolize_names: true)
        expect(content).to have_key(:topics)
        expect(content).to have_key(:config)
      ensure
        FileUtils.rm_f(temp_file)
      end
    end
    
    it "includes all configuration parameters" do
      temp_file = "/tmp/test_config_#{Time.now.to_i}.json"
      
      begin
        Topical::ModelSerializer.save(engine, temp_file)
        content = JSON.parse(File.read(temp_file), symbolize_names: true)
        
        config = content[:config]
        expect(config).to have_key(:clustering_method)
        expect(config).to have_key(:reduce_dimensions)
        expect(config).to have_key(:labeling_method)
      ensure
        FileUtils.rm_f(temp_file)
      end
    end
  end
  
  describe ".load" do
    it "loads engine from JSON file" do
      temp_file = "/tmp/test_load_serializer_#{Time.now.to_i}.json"
      
      begin
        # Save first
        Topical::ModelSerializer.save(engine, temp_file)
        original_topics_count = engine.topics.length
        
        # Load
        loaded_engine = Topical::ModelSerializer.load(temp_file)
        
        expect(loaded_engine).to be_a(Topical::Engine)
        expect(loaded_engine.topics.length).to eq(original_topics_count)
      ensure
        FileUtils.rm_f(temp_file)
      end
    end
    
    it "handles missing file gracefully" do
      expect {
        Topical::ModelSerializer.load("/nonexistent/file.json")
      }.to raise_error(Errno::ENOENT)
    end
    
    it "handles malformed JSON gracefully" do
      temp_file = "/tmp/test_malformed_serializer_#{Time.now.to_i}.json"
      
      begin
        File.write(temp_file, "{ invalid json }")
        expect {
          Topical::ModelSerializer.load(temp_file)
        }.to raise_error(JSON::ParserError)
      ensure
        FileUtils.rm_f(temp_file)
      end
    end
  end
  
  it "preserves engine state through save/load cycle" do
    temp_file = "/tmp/test_roundtrip_serializer_#{Time.now.to_i}.json"
    
    begin
      # Save and load
      Topical::ModelSerializer.save(engine, temp_file)
      loaded_engine = Topical::ModelSerializer.load(temp_file)
      
      # Compare essential properties
      expect(loaded_engine.topics.length).to eq(engine.topics.length)
      expect(loaded_engine.topics.map(&:id).sort).to eq(engine.topics.map(&:id).sort)
      expect(loaded_engine.clustering_adapter.class).to eq(engine.clustering_adapter.class)
    ensure
      FileUtils.rm_f(temp_file)
    end
  end
end