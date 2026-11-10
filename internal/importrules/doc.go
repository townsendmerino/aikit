// Package importrules holds no code. Its test pins the core module's import graph:
// which aikit packages each package imports, which external modules each package
// reaches, and that nothing in the core reaches gpu/ or chunk/treesitter. It is the
// check behind design rule 2 in docs/architecture.md ("packages are small leaves; the
// DAG stays shallow"), which the package diagram there is drawn from.
package importrules
