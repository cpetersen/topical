# frozen_string_literal: true

module Topical
  module Labelers
    # Fast term-based labeling using top distinctive terms
    class TermBased < Base
      def generate_label(topic)
        terms = topic.terms
        return "Topic #{topic.id}" if terms.empty?
        
        # Take top distinctive terms
        label_terms = terms.first(3).select { |t| t.length > 3 }
        
        if label_terms.length >= 2
          "#{capitalize_phrase(label_terms[0])} & #{capitalize_phrase(label_terms[1])}"
        else
          capitalize_phrase(label_terms.first || terms.first)
        end
      end
    end
  end
end