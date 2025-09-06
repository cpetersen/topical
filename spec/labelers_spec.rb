# frozen_string_literal: true

require 'spec_helper'

RSpec.describe "Labelers" do
  # Mock topic object for testing
  let(:mock_topic) do
    double("Topic",
      id: 0,
      terms: ["machine", "learning", "neural", "network", "deep"],
      documents: [
        "Machine learning with neural networks for classification",
        "Deep learning algorithms and neural network architectures", 
        "Training neural networks on large datasets"
      ]
    ).tap do |topic|
      # Add representative_docs method
      allow(topic).to receive(:representative_docs) do |k: 3|
        topic.documents.first(k)
      end
    end
  end

  let(:empty_topic) do
    double("Topic", id: 1, terms: [], documents: ["Empty topic"])
  end

  let(:short_terms_topic) do
    double("Topic", id: 2, terms: ["ai", "ml", "nn"], documents: ["AI and ML"])
  end

  describe Topical::Labelers::Base do
    let(:base_labeler) { Topical::Labelers::Base.new }

    describe "#generate_label" do
      it "raises NotImplementedError" do
        expect {
          base_labeler.generate_label(mock_topic)
        }.to raise_error(NotImplementedError, "Subclasses must implement generate_label")
      end
    end

    describe "#capitalize_phrase" do
      it "capitalizes words separated by spaces" do
        result = base_labeler.send(:capitalize_phrase, "machine learning")
        expect(result).to eq("Machine Learning")
      end

      it "capitalizes words separated by underscores" do
        result = base_labeler.send(:capitalize_phrase, "neural_network")
        expect(result).to eq("Neural Network")
      end

      it "capitalizes words separated by hyphens" do
        result = base_labeler.send(:capitalize_phrase, "deep-learning")
        expect(result).to eq("Deep Learning")
      end

      it "handles single words" do
        result = base_labeler.send(:capitalize_phrase, "classification")
        expect(result).to eq("Classification")
      end
    end

    describe "#select_representative_docs" do
      let(:documents) { ["doc1", "doc2", "doc3", "doc4", "doc5"] }

      it "returns all documents when count <= k" do
        result = base_labeler.send(:select_representative_docs, documents.first(3), k: 5)
        expect(result).to eq(documents.first(3))
      end

      it "returns first k documents when count > k" do
        result = base_labeler.send(:select_representative_docs, documents, k: 3)
        expect(result).to eq(documents.first(3))
      end
    end
  end

  describe Topical::Labelers::TermBased do
    let(:term_labeler) { Topical::Labelers::TermBased.new }

    describe "#generate_label" do
      it "generates label from top 2 terms when available" do
        label = term_labeler.generate_label(mock_topic)
        expect(label).to eq("Machine & Learning")
      end

      it "generates single term label when only one long term" do
        single_term_topic = double("Topic", 
          id: 3, 
          terms: ["classification", "ai", "ml"]
        )
        
        label = term_labeler.generate_label(single_term_topic)
        expect(label).to eq("Classification")
      end

      it "falls back to first term when no long terms" do
        label = term_labeler.generate_label(short_terms_topic)
        expect(label).to eq("Ai")
      end

      it "returns default label for empty terms" do
        label = term_labeler.generate_label(empty_topic)
        expect(label).to eq("Topic 1")
      end

      it "filters out short terms (length <= 3)" do
        mixed_topic = double("Topic", 
          id: 4, 
          terms: ["ai", "machine", "learning", "ml", "classification"]
        )
        
        label = term_labeler.generate_label(mixed_topic)
        expect(label).to eq("Machine & Learning")
      end

      it "handles topic with only one long term" do
        one_long_term_topic = double("Topic",
          id: 5,
          terms: ["ai", "ml", "classification"]
        )
        
        label = term_labeler.generate_label(one_long_term_topic)
        expect(label).to eq("Classification")
      end
    end
  end

  describe Topical::Labelers::LLMBased do
    let(:mock_provider) do
      double("LLMProvider").tap do |provider|
        allow(provider).to receive(:available?).and_return(true)
        allow(provider).to receive(:generate).and_return(
          '{"label": "Machine Learning Systems", "description": "Topic about ML", "confidence": 0.9}'
        )
      end
    end

    let(:unavailable_provider) do
      double("UnavailableProvider").tap do |provider|
        allow(provider).to receive(:available?).and_return(false)
      end
    end

    describe "#generate_label with provider" do
      it "uses LLM when provider is available" do
        llm_labeler = Topical::Labelers::LLMBased.new(provider: mock_provider)
        
        label = llm_labeler.generate_label(mock_topic)
        expect(label).to eq("Machine Learning Systems")
        
        expect(mock_provider).to have_received(:generate).with(
          prompt: anything,
          max_tokens: 150,
          temperature: 0.3,
          response_format: { type: "json_object" }
        )
      end

      it "falls back to term-based when LLM fails" do
        failing_provider = double("FailingProvider")
        allow(failing_provider).to receive(:available?).and_return(true)
        allow(failing_provider).to receive(:generate).and_raise(StandardError, "API Error")
        
        llm_labeler = Topical::Labelers::LLMBased.new(provider: failing_provider)
        
        # Should fall back to term-based labeling
        label = llm_labeler.generate_label(mock_topic)
        expect(label).to eq("Machine & Learning") # TermBased result
      end

      it "falls back to term-based when provider unavailable" do
        # Mock llm_available? to return false
        llm_labeler = Topical::Labelers::LLMBased.new(provider: unavailable_provider)
        allow(llm_labeler).to receive(:llm_available?).and_return(false)
        
        label = llm_labeler.generate_label(mock_topic)
        expect(label).to eq("Machine & Learning") # TermBased result
      end
    end

    describe "#generate_label without provider" do
      it "falls back to term-based when no LLM available" do
        llm_labeler = Topical::Labelers::LLMBased.new
        
        # Mock the LLMProvider.default to return nil (no LLM available)
        allow(Topical::Labelers::LLMProvider).to receive(:default).and_return(nil)
        
        label = llm_labeler.generate_label(mock_topic)
        expect(label).to eq("Machine & Learning") # TermBased result
      end

      it "handles LoadError when requiring LLM adapter" do
        llm_labeler = Topical::Labelers::LLMBased.new
        
        # Mock require to raise LoadError
        allow(llm_labeler).to receive(:require_relative).and_raise(LoadError)
        
        label = llm_labeler.generate_label(mock_topic)
        expect(label).to eq("Machine & Learning") # TermBased result
      end
    end

    describe "private methods" do
      let(:llm_labeler) { Topical::Labelers::LLMBased.new(provider: mock_provider) }

      describe "#build_analysis_prompt" do
        it "includes distinctive terms and sample documents" do
          documents = ["Sample doc 1", "Sample doc 2"]
          terms = ["machine", "learning", "neural"]
          
          prompt = llm_labeler.send(:build_analysis_prompt, documents, terms)
          
          expect(prompt).to include("machine, learning, neural")
          expect(prompt).to include("Document 1:")
          expect(prompt).to include("Sample doc 1")
          expect(prompt).to include("Document 2:")
          expect(prompt).to include("Sample doc 2")
          expect(prompt).to include("JSON response")
        end

        it "truncates long documents" do
          long_doc = "x" * 400
          documents = [long_doc]
          terms = ["test"]
          
          prompt = llm_labeler.send(:build_analysis_prompt, documents, terms)
          
          expect(prompt).to include("x" * 300 + "...")
          expect(prompt).not_to include("x" * 400)
        end
      end

      describe "#clean_label" do
        it "removes quotes and trims whitespace" do
          result = llm_labeler.send(:clean_label, '"Machine Learning"')
          expect(result).to eq("Machine Learning")
        end

        it "takes first line for multiline labels" do
          result = llm_labeler.send(:clean_label, "Machine Learning\nExtra text")
          expect(result).to eq("Machine Learning")
        end

        it "truncates very long labels" do
          long_label = "x" * 60
          result = llm_labeler.send(:clean_label, long_label)
          expect(result).to eq("x" * 48 + "...")
        end

        it "returns default for nil/empty labels" do
          expect(llm_labeler.send(:clean_label, nil)).to eq("Unknown Topic")
          expect(llm_labeler.send(:clean_label, "")).to eq("")
          expect(llm_labeler.send(:clean_label, "   ")).to eq("")
        end
      end

      describe "#analyze_with_llm" do
        it "parses valid JSON response" do
          json_response = '{"label": "AI Systems", "description": "About AI", "confidence": 0.8}'
          allow(mock_provider).to receive(:generate).and_return(json_response)
          
          result = llm_labeler.send(:analyze_with_llm, ["doc"], ["term"])
          
          expect(result).to include(
            label: "AI Systems",
            description: "About AI", 
            confidence: 0.8
          )
        end

        it "handles malformed JSON gracefully" do
          allow(mock_provider).to receive(:generate).and_return("Invalid JSON")
          
          expect {
            llm_labeler.send(:analyze_with_llm, ["doc"], ["term"])
          }.to raise_error(JSON::ParserError)
        end
      end
    end
  end

  describe Topical::Labelers::Hybrid do
    let(:mock_llm_provider) do
      double("LLMProvider").tap do |provider|
        allow(provider).to receive(:available?).and_return(true)
        allow(provider).to receive(:generate).and_return(
          '{"label": "Advanced ML", "description": "Enhanced topic", "confidence": 0.9}'
        )
      end
    end

    describe "#generate_label" do
      it "uses LLM result when available and different" do
        hybrid_labeler = Topical::Labelers::Hybrid.new(provider: mock_llm_provider)
        
        label = hybrid_labeler.generate_label(mock_topic)
        expect(label).to eq("Advanced ML")
      end

      it "falls back to term-based when LLM unavailable" do
        unavailable_provider = double("UnavailableProvider")
        allow(unavailable_provider).to receive(:available?).and_return(false)
        
        hybrid_labeler = Topical::Labelers::Hybrid.new(provider: unavailable_provider)
        
        # Mock the LLMBased labeler to fall back to term-based
        allow_any_instance_of(Topical::Labelers::LLMBased).to receive(:llm_available?).and_return(false)
        
        label = hybrid_labeler.generate_label(mock_topic)
        expect(label).to eq("Machine & Learning") # TermBased result
      end

      it "uses term-based when LLM returns default pattern" do
        # Mock LLM to return a pattern that indicates fallback
        failing_provider = double("FailingProvider")
        allow(failing_provider).to receive(:available?).and_return(false)
        
        hybrid_labeler = Topical::Labelers::Hybrid.new(provider: failing_provider)
        
        # Mock the LLMBased labeler to fall back to term-based
        allow_any_instance_of(Topical::Labelers::LLMBased).to receive(:llm_available?).and_return(false)
        
        # This will cause LLMBased to fall back to TermBased
        label = hybrid_labeler.generate_label(mock_topic)
        expect(label).to eq("Machine & Learning")
      end
    end
  end
end