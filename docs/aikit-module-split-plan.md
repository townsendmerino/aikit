# aikit ⇄ goinfer split — moved

The full, up-to-date execution plan for splitting the codebase into the stable
`aikit` retrieval toolkit and the separately-paced `goinfer` LLM runtime now
lives in the **goinfer** repo (it's driven from there in VSCode):

> `goinfer/docs/migration-plan.md`
> — https://github.com/townsendmerino/goinfer/blob/main/docs/migration-plan.md

That document is the source of truth: target module layout, the `encoder`/`decoder`
`Backend` inversion that quarantines cgo `webgpu`, the `chunk/treesitter` submodule
that quarantines `gotreesitter`, execution order, versioning, CI, and the
definition of done.

The original 1.0 critique that motivated this work stays here:
[`road-to-1.0-critique.md`](./road-to-1.0-critique.md).
