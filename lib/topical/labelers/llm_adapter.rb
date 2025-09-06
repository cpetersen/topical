# frozen_string_literal: true

module Topical
  module Labelers
    # Adapter to allow different LLM backends (red-candle, remote APIs, etc.)
    class LLMAdapter
      # Factory method to create appropriate LLM client
      def self.create(type: :auto, **options)
        case type
        when :red_candle
          RedCandleAdapter.new(**options)
        when :openai
          # Future: OpenAIAdapter.new(**options)
          raise NotImplementedError, "OpenAI adapter not yet implemented"
        when :anthropic
          # Future: AnthropicAdapter.new(**options)
          raise NotImplementedError, "Anthropic adapter not yet implemented"
        when :auto
          # Try red-candle first, then fall back to others
          begin
            RedCandleAdapter.new(**options)
          rescue LoadError
            nil  # No LLM available
          end
        else
          raise ArgumentError, "Unknown LLM type: #{type}"
        end
      end
    end
    
    # Adapter for red-candle (local LLMs)
    class RedCandleAdapter
      def initialize(model: nil, **options)
        require 'red-candle'
        
        @model = model || default_model
        @options = options
        @llm = load_or_create_llm
      end
      
      def generate(prompt:, max_tokens: 100, temperature: 0.3, response_format: nil)
        # Red-candle specific generation
        response = @llm.generate(
          prompt,
          max_length: max_tokens,
          temperature: temperature,
          do_sample: temperature > 0
        )
        
        # Handle JSON response format if requested
        if response_format && response_format[:type] == "json_object"
          ensure_json_response(response)
        else
          response
        end
      end
      
      def available?
        true
      end
      
      private
      
      def default_model
        # Use a small, fast model by default for topic labeling
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
      end
      
      def load_or_create_llm
        # Create new LLM instance with red-candle
        RedCandle::Model.new(
          model_id: @model,
          model_type: :llama,
          quantized: true
        )
      end
      
      def ensure_json_response(response)
        # Try to extract JSON from response
        begin
          require 'json'
          # Look for JSON-like content
          json_match = response.match(/\{.*\}/m)
          if json_match
            JSON.parse(json_match[0])
            json_match[0]  # Return the JSON string if valid
          else
            # Generate a basic JSON response
            generate_fallback_json(response)
          end
        rescue JSON::ParserError
          generate_fallback_json(response)
        end
      end
      
      def generate_fallback_json(text)
        # Create a simple JSON from text response
        require 'json'
        label = text.lines.first&.strip || "Unknown"
        {
          label: label,
          description: text,
          confidence: 0.5
        }.to_json
      end
    end
    
    # Future adapter for remote LLMs
    class RemoteAdapter
      def initialize(api_key:, endpoint:, **options)
        @api_key = api_key
        @endpoint = endpoint
        @options = options
      end
      
      def generate(prompt:, max_tokens: 100, temperature: 0.3, response_format: nil)
        # Make API call
        raise NotImplementedError, "Remote LLM adapter coming soon"
      end
      
      def available?
        !@api_key.nil?
      end
    end
  end
end
