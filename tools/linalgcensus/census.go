package main

import (
	"go/ast"
	"go/token"
	"go/types"
	"path/filepath"
	"strings"
)

// How a file reaches a name.
const (
	named = "named" // an identifier resolves to it (a call, a selector, a type name, a composite-literal key)
	value = "value" // an expression has its type, though no identifier names it (a type reached through a return value)
	iface = "iface" // a method an interface declared in the file could dispatch to, when a linalg type satisfies it
)

// Name is one exported name of the target package.
type Name struct {
	Key     string          // "F", "T", "T.M" (method) or "T.F" (field)
	Kind    string          // func, type, const, var, method, field
	Decl    string          // declaring file, relative to the aikit root
	Configs map[string]bool // configurations it is declared in
}

// use is one file reaching one name.
type use struct{ key, file, how string }

// census accumulates over configurations.
type census struct {
	target string // import path of the target package
	names  map[string]*Name
	uses   map[use]map[string]bool // configurations each use was seen in
	errs   map[string][]string     // type errors by package and configuration, ours only
}

func newCensus(target string) *census {
	return &census{target: target, names: map[string]*Name{}, uses: map[use]map[string]bool{}, errs: map[string][]string{}}
}

// keyOf names a target-package object as the census keys it, or "" when it is not an exported package-level name, an
// exported method of an exported type, or an exported field of an exported struct type.
func (c *census) keyOf(obj types.Object, fields map[*types.Var]string) string {
	if obj == nil || obj.Pkg() == nil || obj.Pkg().Path() != c.target || !obj.Exported() {
		return ""
	}
	switch o := obj.(type) {
	case *types.Func:
		o = o.Origin()
		if r := o.Signature().Recv(); r != nil {
			t := recvName(r.Type())
			if t == "" || !ast.IsExported(t) {
				return ""
			}
			return t + "." + o.Name()
		}
		if o.Parent() == o.Pkg().Scope() {
			return o.Name()
		}
	case *types.Var:
		if o.IsField() {
			return fields[o.Origin()]
		}
		if o.Parent() == o.Pkg().Scope() {
			return o.Name()
		}
	case *types.Const, *types.TypeName:
		if obj.Parent() == obj.Pkg().Scope() {
			return obj.Name()
		}
	}
	return ""
}

func recvName(t types.Type) string {
	if p, ok := t.(*types.Pointer); ok {
		t = p.Elem()
	}
	switch n := types.Unalias(t).(type) {
	case *types.Named:
		return n.Origin().Obj().Name()
	}
	return ""
}

// fieldKeys maps each exported field of an exported struct type in pkg to its key. Test variants of the target are
// distinct packages with distinct field objects, so this runs per variant.
func fieldKeys(pkg *types.Package) map[*types.Var]string {
	m := map[*types.Var]string{}
	sc := pkg.Scope()
	for _, n := range sc.Names() {
		tn, ok := sc.Lookup(n).(*types.TypeName)
		if !ok || !tn.Exported() {
			continue
		}
		if st, ok := tn.Type().Underlying().(*types.Struct); ok {
			for f := range st.Fields() {
				if f.Exported() {
					m[f] = n + "." + f.Name()
				}
			}
		}
	}
	return m
}

// declare records the target's exported names from its plain (non-test) package in one configuration.
func (c *census) declare(w *world, pkg *types.Package, aikitRoot, cfg string) {
	add := func(key, kind string, obj types.Object) {
		n := c.names[key]
		if n == nil {
			n = &Name{Key: key, Kind: kind, Configs: map[string]bool{}}
			if p := w.fset.Position(obj.Pos()); p.Filename != "" {
				if rel, err := filepath.Rel(aikitRoot, p.Filename); err == nil {
					n.Decl = filepath.ToSlash(rel)
				}
			}
			c.names[key] = n
		}
		n.Configs[cfg] = true
	}
	sc := pkg.Scope()
	for _, nm := range sc.Names() {
		obj := sc.Lookup(nm)
		if !obj.Exported() || strings.HasSuffix(w.fset.Position(obj.Pos()).Filename, "_test.go") {
			continue
		}
		switch o := obj.(type) {
		case *types.Func:
			add(nm, "func", o)
		case *types.Const:
			add(nm, "const", o)
		case *types.Var:
			add(nm, "var", o)
		case *types.TypeName:
			add(nm, "type", o)
			if named, ok := types.Unalias(o.Type()).(*types.Named); ok {
				for m := range named.Methods() {
					if m.Exported() {
						add(nm+"."+m.Name(), "method", m)
					}
				}
			}
			if st, ok := o.Type().Underlying().(*types.Struct); ok {
				for f := range st.Fields() {
					if f.Exported() {
						add(nm+"."+f.Name(), "field", f)
					}
				}
			}
		}
	}
}

// record walks one of our packages' use and type records. A test variant contributes only its _test.go files; the
// non-test files it repeats are recorded from the plain package.
func (c *census) record(w *world, l *loaded, plainTarget *types.Package, cfg string) {
	if l.info == nil {
		return
	}
	targetFields := map[*types.Package]map[*types.Var]string{}
	fieldsOf := func(p *types.Package) map[*types.Var]string {
		if targetFields[p] == nil {
			targetFields[p] = fieldKeys(p)
		}
		return targetFields[p]
	}
	keep := func(file string) bool {
		test := strings.HasSuffix(file, "_test.go")
		return (l.ForTest == "") != test
	}
	add := func(key, file, how string) {
		if key == "" || !keep(file) {
			return
		}
		u := use{key, file, how}
		if c.uses[u] == nil {
			c.uses[u] = map[string]bool{}
		}
		c.uses[u][cfg] = true
	}
	for id, obj := range l.info.Uses {
		if obj == nil || obj.Pkg() == nil || obj.Pkg().Path() != c.target {
			continue
		}
		add(c.keyOf(obj, fieldsOf(obj.Pkg())), w.fset.Position(id.Pos()).Filename, named)
	}
	for e, tv := range l.info.Types {
		if tv.IsType() || tv.Type == nil {
			continue
		}
		if k := c.typeKey(tv.Type); k != "" {
			add(k, w.fset.Position(e.Pos()).Filename, value)
		}
	}
	if plainTarget == nil {
		return
	}
	// An interface declared here that a target type satisfies could dispatch to that type's methods: a named
	// interface type, or an interface literal written in place (a parameter's interface{ M() }), which is how a
	// test often takes a *linalg.WeightMat without naming it.
	sc := plainTarget.Scope()
	type ifaceAt struct {
		it  *types.Interface
		pos token.Pos
	}
	var ifaces []ifaceAt
	for id, obj := range l.info.Defs {
		if tn, ok := obj.(*types.TypeName); ok {
			if it, ok := tn.Type().Underlying().(*types.Interface); ok {
				ifaces = append(ifaces, ifaceAt{it, id.Pos()})
			}
		}
	}
	for e, tv := range l.info.Types {
		if _, lit := e.(*ast.InterfaceType); lit && tv.IsType() {
			if it, ok := tv.Type.Underlying().(*types.Interface); ok {
				ifaces = append(ifaces, ifaceAt{it, e.Pos()})
			}
		}
	}
	for _, ia := range ifaces {
		it := ia.it
		if it.NumMethods() == 0 {
			continue
		}
		for _, nm := range sc.Names() {
			ttn, ok := sc.Lookup(nm).(*types.TypeName)
			if !ok || !ttn.Exported() || types.IsInterface(ttn.Type()) {
				continue
			}
			for _, t := range []types.Type{ttn.Type(), types.NewPointer(ttn.Type())} {
				if !types.Implements(t, it) {
					continue
				}
				for m := range it.Methods() {
					add(nm+"."+m.Name(), w.fset.Position(ia.pos).Filename, iface)
				}
				break
			}
		}
	}
}

// typeKey names the exported target type an expression's type is, or points to, or holds as elements.
func (c *census) typeKey(t types.Type) string {
	for range 4 {
		switch u := types.Unalias(t).(type) {
		case *types.Pointer:
			t = u.Elem()
			continue
		case *types.Slice:
			t = u.Elem()
			continue
		case *types.Array:
			t = u.Elem()
			continue
		case *types.Named:
			o := u.Origin().Obj()
			if o.Pkg() != nil && o.Pkg().Path() == c.target && o.Exported() && o.Parent() == o.Pkg().Scope() {
				return o.Name()
			}
		}
		return ""
	}
	return ""
}
