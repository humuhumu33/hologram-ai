@row:prism-canonical-manifest @stage:S2 @status:build @executor:rust @lane:default
Feature: Prism canonical manifest, baked and admitted
  Every compiled archive can carry its formal contract: the canonical
  `hologram-ai/archive/1` manifest (prism-archive) describing tensors, κ
  weight bindings, the component graph, and content-addressed provenance,
  baked as an open extension section like the tokenizer. The manifest's κ
  is `blake3:<hex>` of its exact canonical bytes — compile equivalence
  becomes κ equality. Admission is strict decode → index normalization →
  a validator conjunction whose bodies are extracted from a
  kernel-verified Lean module (empty observed axiom sets, attested in the
  prism-contract oracle's repository). The runner refuses an archive
  whose baked manifest does not admit; archives without the section load
  unchanged.

  Scenario: the compiled archive carries an admitted canonical manifest
    Given the handshake-tiny config and its Llama k-form manifest
    When the k-form graph is compiled with the prism manifest baked, twice
    Then both compiles bake byte-identical manifests with one stable prism kappa
    And the baked manifest strictly decodes and admits through the extracted validators
    And a runner loads the manifest-bearing archive but refuses a tampered manifest
