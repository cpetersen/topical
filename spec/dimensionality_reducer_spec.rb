# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Topical::DimensionalityReducer do
  let(:reducer) { Topical::DimensionalityReducer.new(n_components: 10) }
  let(:high_dim_embeddings) { Array.new(5) { Array.new(50) { rand } } }
  let(:low_dim_embeddings) { Array.new(5) { Array.new(5) { rand } } }

  describe "#reduce" do
    it "returns original embeddings if empty" do
      result = reducer.reduce([])
      expect(result).to eq([])
    end
    
    it "returns original embeddings if already low dimensional" do
      result = reducer.reduce(low_dim_embeddings)
      expect(result).to eq(low_dim_embeddings)
    end
    
    it "returns original embeddings when ClusterKit is not available" do
      allow(reducer).to receive(:require).with('clusterkit').and_raise(LoadError)
      
      result = reducer.reduce(high_dim_embeddings)
      expect(result).to eq(high_dim_embeddings)
    end
    
    it "handles reduction errors gracefully" do
      allow(reducer).to receive(:require).with('clusterkit').and_raise(StandardError, "UMAP failed")
      
      result = reducer.reduce(high_dim_embeddings)
      expect(result).to eq(high_dim_embeddings)
    end
  end
  
  describe "#validate_embeddings_for_umap" do
    it "identifies valid embeddings" do
      embeddings = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ]
      
      valid, invalid_indices = reducer.send(:validate_embeddings_for_umap, embeddings)
      
      expect(valid).to eq(embeddings)
      expect(invalid_indices).to be_empty
    end
    
    it "identifies invalid embeddings with NaN" do
      embeddings = [
        [1.0, 2.0, 3.0],
        [Float::NAN, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ]
      
      valid, invalid_indices = reducer.send(:validate_embeddings_for_umap, embeddings)
      
      expect(valid).to eq([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      expect(invalid_indices).to eq([1])
    end
    
    it "identifies invalid embeddings with Infinity" do
      embeddings = [
        [1.0, Float::INFINITY, 3.0],
        [4.0, 5.0, 6.0]
      ]
      
      valid, invalid_indices = reducer.send(:validate_embeddings_for_umap, embeddings)
      
      expect(valid).to eq([[4.0, 5.0, 6.0]])
      expect(invalid_indices).to eq([0])
    end
    
    it "handles non-array embeddings" do
      embeddings = [
        [1.0, 2.0, 3.0],
        "not an array",
        [4.0, 5.0, 6.0]
      ]
      
      valid, invalid_indices = reducer.send(:validate_embeddings_for_umap, embeddings)
      
      expect(valid).to eq([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      expect(invalid_indices).to eq([1])
    end
  end
end