# frozen_string_literal: true

module Topical
  module Labelers
    # LLM-powered topic labeling (requires red-candle or other LLM provider)
    class LLMBased < Base
      def initialize(provider: nil)
        @provider = provider
      end
      
      def generate_label(topic)
        unless llm_available?
          # Fallback to term-based if LLM not available
          return TermBased.new.generate_label(topic)
        end
        
        # Select best documents to send to LLM
        sample_docs = topic.representative_docs(k: 3)
        
        # Generate comprehensive analysis
        response = analyze_with_llm(sample_docs, topic.terms)
        
        response[:label]
      rescue => e
        # Fallback on error
        puts "LLM labeling failed: #{e.message}" if ENV['DEBUG']
        TermBased.new.generate_label(topic)
      end
      
      private
      
      def llm_available?
        return true if @provider
        
        # Try to create LLM adapter
        begin
          require_relative 'llm_adapter'
          @provider = LLMAdapter.create(type: :auto)
          @provider && @provider.available?
        rescue LoadError, StandardError => e
          puts "LLM not available: #{e.message}" if ENV['DEBUG']
          false
        end
      end
      
      def analyze_with_llm(documents, terms)
        prompt = build_analysis_prompt(documents, terms)
        
        response = @provider.generate(
          prompt: prompt,
          max_tokens: 150,
          temperature: 0.3,
          response_format: { type: "json_object" }
        )
        
        # Parse JSON response
        require 'json'
        result = JSON.parse(response, symbolize_names: true)
        
        # Validate and clean
        {
          label: clean_label(result[:label]),
          description: result[:description] || "Topic about #{result[:label]}",
          themes: result[:themes] || [],
          confidence: result[:confidence] || 0.8
        }
      end
      
      def build_analysis_prompt(documents, terms)
        doc_samples = documents.map.with_index do |doc, i|
          preview = doc.length > 300 ? "#{doc[0..300]}..." : doc
          "Document #{i + 1}:\n#{preview}"
        end.join("\n\n")
        
        <<~PROMPT
          Analyze this cluster of related documents and provide a structured summary.
          
          Distinctive terms found: #{terms.first(10).join(', ')}
          
          Sample documents:
          #{doc_samples}
          
          Provide a JSON response with:
          {
            "label": "A 2-4 word topic label",
            "description": "One sentence describing what connects these documents",
            "themes": ["theme1", "theme2", "theme3"],
            "confidence": 0.0-1.0 score of how coherent this topic is
          }
          
          Focus on what meaningfully connects these documents, not just common words.
        PROMPT
      end
      
      def clean_label(label)
        return "Unknown Topic" unless label
        
        # Remove quotes, trim, limit length
        cleaned = label.to_s.strip.gsub(/^["']|["']$/, '')
        cleaned = cleaned.split("\n").first if cleaned.include?("\n")
        
        # Limit to reasonable length
        if cleaned.length > 50
          cleaned[0..47] + "..."
        else
          cleaned
        end
      end
    end
  end
end
