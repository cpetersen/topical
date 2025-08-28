# frozen_string_literal: true

module Topical
  module Labelers
    # Base class for topic labeling strategies
    class Base
      def generate_label(topic)
        raise NotImplementedError, "Subclasses must implement generate_label"
      end
      
      protected
      
      def capitalize_phrase(phrase)
        phrase.split(/[\s_-]/).map(&:capitalize).join(' ')
      end
      
      def select_representative_docs(documents, k: 3)
        return documents if documents.length <= k
        documents.first(k)
      end
    end
  end
end