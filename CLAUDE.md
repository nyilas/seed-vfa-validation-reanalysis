## Project implementation spec

For this project, follow the implementation specification in:

`AGENTS.md`

`AGENTS.md` is the authoritative implementation plan for the Seed/VFA validation-aware reanalysis project.

When building or modifying the project:
- follow the repository layout defined in `AGENTS.md`;
- implement only the v1 scope described there;
- do not add active learning, synthetic augmentation, or GPR uncertainty loops;
- respect all hard invariants about de-duplication, grouped splits, no leakage, deterministic seeds, and fold-by-fold predictions;
- keep the project reproducible and test-driven.

## Additional instruction files

Before editing code, follow:

`rules/python_research.md`

Before editing papers, reports, abstracts, documentation, or scientific text, follow:

`rules/scientific_writing.md`
