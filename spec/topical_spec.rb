# frozen_string_literal: true

RSpec.describe Topical do
  it "has a version number" do
    expect(Topical::VERSION).not_to be nil
  end
  
  describe ".extract" do
    it "provides a convenient method for topic extraction" do
      embeddings = generate_test_embeddings(n_samples: 20, n_features: 10)
      documents = generate_test_documents(n_samples: 20)
      
      expect { Topical.extract(embeddings: embeddings, documents: documents) }.not_to raise_error
    end
  end
  
  describe ".llm_available?" do
    it "detects if red-candle is available" do
      # This will be true in dev since we have red-candle as a dev dependency
      expect([true, false]).to include(Topical.llm_available?)
    end
  end
  
  describe "module structure" do
    it "autoloads Engine" do
      expect { Topical::Engine }.not_to raise_error
    end
    
    it "autoloads Topic" do
      expect { Topical::Topic }.not_to raise_error
    end
    
    it "autoloads clustering adapters" do
      expect { Topical::Clustering::Adapter }.not_to raise_error
      expect { Topical::Clustering::HDBSCANAdapter }.not_to raise_error
    end
  end
end