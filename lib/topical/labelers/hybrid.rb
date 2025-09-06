# frozen_string_literal: true

module Topical
  module Labelers
    # Hybrid labeling that combines term-based and LLM approaches
    class Hybrid < Base
      def initialize(provider: nil, logger: nil)
        @term_labeler = TermBased.new
        @llm_labeler = LLMBased.new(provider: provider, logger: logger)
      end
      
      def generate_label(topic)
        # Start with term-based label
        term_label = @term_labeler.generate_label(topic)
        
        # Try to enhance with LLM if available
        llm_label = @llm_labeler.generate_label(topic)
        
        # For now, just return the LLM label if different, otherwise term label
        llm_label != "LLM Topic #{topic.id}" ? llm_label : term_label
      end
    end
  end
end
