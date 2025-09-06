# frozen_string_literal: true

require 'spec_helper'
require_relative '../lib/topical/labelers/red_candle_adapter'

RSpec.describe "LLM Adapters" do
  describe Topical::Labelers::LLMProvider do
    describe ".default" do
      it "returns red-candle adapter when available" do
        # Mock RedCandleAdapter to succeed
        mock_adapter = double("RedCandleAdapter")
        allow(Topical::Labelers::RedCandleAdapter).to receive(:new).and_return(mock_adapter)
        
        result = Topical::Labelers::LLMProvider.default
        expect(result).to eq(mock_adapter)
      end
      
      it "returns nil when red-candle unavailable" do
        # Mock RedCandleAdapter to raise LoadError
        allow(Topical::Labelers::RedCandleAdapter).to receive(:new).and_raise(LoadError)
        
        result = Topical::Labelers::LLMProvider.default
        expect(result).to be_nil
      end
    end
  end

  describe Topical::Labelers::RedCandleAdapter do
    let(:mock_llm) do
      double("RedCandleLLM").tap do |llm|
        allow(llm).to receive(:generate).and_return("Sample AI response")
      end
    end

    before do
      # Mock red-candle dependency
      stub_const("RedCandle", Module.new)
      stub_const("RedCandle::Model", Class.new do
        def initialize(**options); end
        def generate(*args); "Mock response"; end
      end)
      
      # Allow RedCandleAdapter to be instantiated without errors
      allow_any_instance_of(Topical::Labelers::RedCandleAdapter).to receive(:require).and_call_original
      allow_any_instance_of(Topical::Labelers::RedCandleAdapter).to receive(:require).with('red-candle')
      allow(RedCandle::Model).to receive(:new).and_return(mock_llm)
    end

    describe "#initialize" do
      it "creates adapter with default model" do
        adapter = Topical::Labelers::RedCandleAdapter.new
        expect(adapter).to be_available
      end
      
      it "accepts custom model" do
        custom_model = "custom/model"
        adapter = Topical::Labelers::RedCandleAdapter.new(model: custom_model)
        
        expect(RedCandle::Model).to have_received(:new).with(
          model_id: custom_model,
          model_type: :llama,
          quantized: true
        )
      end
    end

    describe "#generate" do
      let(:adapter) { Topical::Labelers::RedCandleAdapter.new }
      
      it "calls LLM with correct parameters" do
        prompt = "Test prompt"
        
        result = adapter.generate(
          prompt: prompt,
          max_tokens: 50,
          temperature: 0.5
        )
        
        expect(mock_llm).to have_received(:generate).with(
          prompt,
          max_length: 50,
          temperature: 0.5,
          do_sample: true
        )
      end
      
      it "sets do_sample to false when temperature is 0" do
        adapter.generate(
          prompt: "test",
          temperature: 0
        )
        
        expect(mock_llm).to have_received(:generate).with(
          "test",
          max_length: 100, # default
          temperature: 0,
          do_sample: false
        )
      end
      
      context "with JSON response format" do
        it "ensures JSON format when requested" do
          allow(mock_llm).to receive(:generate).and_return('{"key": "value"}')
          
          result = adapter.generate(
            prompt: "test",
            response_format: { type: "json_object" }
          )
          
          expect(result).to eq('{"key": "value"}')
        end
        
        it "extracts JSON from mixed response" do
          allow(mock_llm).to receive(:generate).and_return('Some text {"valid": "json"} more text')
          
          result = adapter.generate(
            prompt: "test",
            response_format: { type: "json_object" }
          )
          
          expect(result).to eq('{"valid": "json"}')
        end
        
        it "generates fallback JSON for invalid responses" do
          allow(mock_llm).to receive(:generate).and_return('Not JSON at all')
          
          result = adapter.generate(
            prompt: "test",
            response_format: { type: "json_object" }
          )
          
          # Should generate fallback JSON
          expect(result).to be_a(String)
          expect { JSON.parse(result) }.not_to raise_error
          
          parsed = JSON.parse(result)
          expect(parsed).to have_key("label")
          expect(parsed).to have_key("description")
          expect(parsed).to have_key("confidence")
        end
      end
    end

    describe "#available?" do
      it "returns true when initialized successfully" do
        adapter = Topical::Labelers::RedCandleAdapter.new
        expect(adapter).to be_available
      end
    end

    describe "private methods" do
      let(:adapter) { Topical::Labelers::RedCandleAdapter.new }
      
      describe "#ensure_json_response" do
        it "returns valid JSON unchanged" do
          valid_json = '{"test": "value"}'
          result = adapter.send(:ensure_json_response, valid_json)
          expect(result).to eq(valid_json)
        end
        
        it "extracts JSON from mixed text" do
          mixed_text = 'Here is the result: {"extracted": "json"} with more text'
          result = adapter.send(:ensure_json_response, mixed_text)
          expect(result).to eq('{"extracted": "json"}')
        end
        
        it "generates fallback for non-JSON text" do
          text = "Just plain text response"
          result = adapter.send(:ensure_json_response, text)
          
          parsed = JSON.parse(result)
          expect(parsed["label"]).to eq("Just plain text response")
          expect(parsed["description"]).to eq(text)
          expect(parsed["confidence"]).to eq(0.5)
        end
      end
      
      describe "#generate_fallback_json" do
        it "creates valid JSON from text" do
          text = "Machine Learning Topic\nWith multiple lines"
          result = adapter.send(:generate_fallback_json, text)
          
          parsed = JSON.parse(result)
          expect(parsed["label"]).to eq("Machine Learning Topic")
          expect(parsed["description"]).to eq(text)
          expect(parsed["confidence"]).to eq(0.5)
        end
        
        it "handles empty text" do
          result = adapter.send(:generate_fallback_json, "")
          parsed = JSON.parse(result)
          expect(parsed["label"]).to eq("Unknown")
        end
      end
    end
  end

end