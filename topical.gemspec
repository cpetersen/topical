# frozen_string_literal: true

require_relative "lib/topical/version"

Gem::Specification.new do |spec|
  spec.name = "topical"
  spec.version = Topical::VERSION
  spec.authors = ["Chris Petersen"]
  spec.email = ["chris@petersen.io"]

  spec.summary = "Topic modeling for Ruby using modern clustering algorithms"
  spec.description = "Extract topics from document embeddings using HDBSCAN clustering and c-TF-IDF term extraction. Provides automatic topic labeling, quality metrics, and support for various clustering algorithms."
  spec.homepage = "https://github.com/cpetersen/topical"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.1.0"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://github.com/cpetersen/topical"
  spec.metadata["changelog_uri"] = "https://github.com/cpetersen/topical/blob/main/CHANGELOG.md"
  spec.metadata["documentation_uri"] = "https://rubydoc.info/gems/topical"

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  gemspec = File.basename(__FILE__)
  spec.files = IO.popen(%w[git ls-files -z], chdir: __dir__, err: IO::NULL) do |ls|
    ls.readlines("\x0", chomp: true).reject do |f|
      (f == gemspec) ||
        f.start_with?(*%w[bin/ test/ spec/ features/ .git appveyor Gemfile])
    end
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  # Core dependencies
  spec.add_dependency "clusterkit", "~> 0.2"

  # Optional dependencies (for enhanced features)
  # Users can add red-candle if they want LLM-powered labeling
  spec.add_development_dependency "red-candle", ">= 1.2"

  # Development dependencies
  spec.add_development_dependency "rspec", "~> 3.0"
  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "standard", "~> 1.3"
end
