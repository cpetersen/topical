# frozen_string_literal: true

require 'set'

module Topical
  module Metrics
    extend self
    
    # Compute UMass Coherence for topic quality
    # Higher coherence = more interpretable topic
    def compute_coherence(terms, documents, top_n: 10)
      return 0.0 if terms.empty? || documents.empty?
      
      # Use top N terms
      eval_terms = terms.first(top_n)
      return 0.0 if eval_terms.length < 2
      
      # Create document term matrix for co-occurrence
      doc_term_counts = count_cooccurrences(eval_terms, documents)
      
      # Compute UMass coherence
      coherence_sum = 0.0
      pairs_count = 0
      
      eval_terms.each_with_index do |term_i, i|
        eval_terms.each_with_index do |term_j, j|
          next unless j < i  # Only upper triangle
          
          # P(term_i, term_j) = co-occurrence count
          cooccur = doc_term_counts["#{term_i},#{term_j}"] || 0
          # P(term_j) = document frequency
          doc_freq_j = doc_term_counts[term_j] || 0
          
          if cooccur > 0 && doc_freq_j > 0
            # UMass: log((cooccur + 1) / doc_freq_j)
            coherence_sum += Math.log((cooccur + 1.0) / doc_freq_j)
            pairs_count += 1
          end
        end
      end
      
      return 0.0 if pairs_count == 0
      
      # Normalize by number of pairs
      coherence = coherence_sum / pairs_count
      
      # Transform to 0-1 range (coherence is typically negative)
      # More negative = less coherent, so we reverse and bound
      normalized = 1.0 / (1.0 + Math.exp(-coherence))
      normalized
    end
    
    # Compute how distinct a topic is from others
    def compute_distinctiveness(topic, other_topics)
      return 1.0 if other_topics.empty?
      
      topic_terms = Set.new(topic.terms.first(20))
      
      # Compare with other topics
      overlaps = other_topics.map do |other|
        next if other.id == topic.id
        
        other_terms = Set.new(other.terms.first(20))
        overlap = (topic_terms & other_terms).size.to_f
        
        # Jaccard similarity
        union_size = (topic_terms | other_terms).size
        union_size > 0 ? overlap / union_size : 0
      end.compact
      
      return 1.0 if overlaps.empty?
      
      # Distinctiveness = 1 - average overlap
      1.0 - (overlaps.sum / overlaps.length)
    end
    
    # Compute diversity across all topics
    def compute_diversity(topics)
      return 0.0 if topics.length < 2
      
      # Collect all term sets
      term_sets = topics.map { |t| Set.new(t.terms.first(20)) }
      
      # Compute pairwise Jaccard distances
      distances = []
      term_sets.each_with_index do |set_i, i|
        term_sets.each_with_index do |set_j, j|
          next unless j > i  # Only upper triangle
          
          intersection = (set_i & set_j).size.to_f
          union = (set_i | set_j).size.to_f
          
          # Jaccard distance = 1 - Jaccard similarity
          distance = union > 0 ? 1.0 - (intersection / union) : 1.0
          distances << distance
        end
      end
      
      # Average distance = diversity
      distances.sum / distances.length
    end
    
    # Compute coverage (what fraction of docs are in topics vs outliers)
    def compute_coverage(topics, total_documents)
      return 0.0 if total_documents == 0
      
      docs_in_topics = topics.sum(&:size)
      docs_in_topics.to_f / total_documents
    end
    
    # Silhouette score for cluster quality
    def compute_silhouette_score(topic, all_topics, embeddings)
      return 0.0 if topic.embeddings.empty?
      
      silhouettes = []
      
      topic.embeddings.each_with_index do |embedding, idx|
        # a(i) = average distance to other points in same cluster
        if topic.embeddings.length > 1
          a_i = topic.embeddings.each_with_index
                    .reject { |_, j| j == idx }
                    .map { |other, _| euclidean_distance(embedding, other) }
                    .sum.to_f / (topic.embeddings.length - 1)
        else
          a_i = 0.0
        end
        
        # b(i) = minimum average distance to points in other clusters
        b_values = all_topics.reject { |t| t.id == topic.id }.map do |other_topic|
          next if other_topic.embeddings.empty?
          
          avg_dist = other_topic.embeddings
                               .map { |other| euclidean_distance(embedding, other) }
                               .sum.to_f / other_topic.embeddings.length
          avg_dist
        end.compact
        
        b_i = b_values.min || a_i
        
        # Silhouette coefficient
        if a_i == 0 && b_i == 0
          s_i = 0
        else
          s_i = (b_i - a_i) / [a_i, b_i].max
        end
        
        silhouettes << s_i
      end
      
      # Average silhouette score for topic
      silhouettes.sum / silhouettes.length
    end
    
    private
    
    def count_cooccurrences(terms, documents)
      counts = Hash.new(0)
      
      documents.each do |doc|
        doc_lower = doc.downcase
        
        # Count individual term occurrences
        terms.each do |term|
          counts[term] += 1 if doc_lower.include?(term.downcase)
        end
        
        # Count co-occurrences
        terms.each_with_index do |term_i, i|
          terms.each_with_index do |term_j, j|
            next unless j < i
            
            if doc_lower.include?(term_i.downcase) && doc_lower.include?(term_j.downcase)
              counts["#{term_i},#{term_j}"] += 1
            end
          end
        end
      end
      
      counts
    end
    
    def euclidean_distance(vec1, vec2)
      Math.sqrt(
        vec1.zip(vec2).map { |a, b| (a - b) ** 2 }.sum
      )
    end
  end
end
