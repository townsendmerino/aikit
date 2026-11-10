package retrievalzone

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func git(root string, args ...string) ([]byte, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return out, nil
}

func splitZ(b []byte) []string {
	var out []string
	for _, s := range strings.Split(string(b), "\x00") {
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

// needsContent is what Analyze reads: Go sources, assembly, and go.mod files.
func needsContent(p string) bool {
	return strings.HasSuffix(p, ".go") || strings.HasSuffix(p, ".s") || filepath.Base(p) == "go.mod"
}

type refSource struct {
	files   []string
	content map[string][]byte
}

func (s *refSource) Files() []string { return s.files }
func (s *refSource) Read(p string) ([]byte, error) {
	b, ok := s.content[p]
	if !ok {
		return nil, fmt.Errorf("%s: not in this tree", p)
	}
	return b, nil
}

// AtRef is the tree at a git ref (a tag, a commit).
func AtRef(root, ref string) (Source, error) {
	out, err := git(root, "ls-tree", "-r", "-z", "--name-only", ref)
	if err != nil {
		return nil, err
	}
	s := &refSource{files: splitZ(out), content: map[string][]byte{}}
	var want []string
	var stdin bytes.Buffer
	for _, f := range s.files {
		if needsContent(f) {
			want = append(want, f)
			fmt.Fprintf(&stdin, "%s:%s\n", ref, f)
		}
	}
	cmd := exec.Command("git", "cat-file", "--batch")
	cmd.Dir = root
	cmd.Stdin = &stdin
	raw, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git cat-file --batch at %s: %v", ref, err)
	}
	r := bufio.NewReader(bytes.NewReader(raw))
	for _, f := range want {
		head, err := r.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("git cat-file at %s: short output at %s", ref, f)
		}
		var sha, typ string
		var size int
		if _, err := fmt.Sscanf(head, "%s %s %d", &sha, &typ, &size); err != nil || typ != "blob" {
			return nil, fmt.Errorf("git cat-file at %s: %s: unexpected header %q", ref, f, strings.TrimSpace(head))
		}
		buf := make([]byte, size+1) // content, then a newline
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("git cat-file at %s: %s: %v", ref, f, err)
		}
		s.content[f] = buf[:size]
	}
	return s, nil
}

type workSource struct {
	root  string
	files []string
}

func (s *workSource) Files() []string { return s.files }
func (s *workSource) Read(p string) ([]byte, error) {
	return os.ReadFile(filepath.Join(s.root, filepath.FromSlash(p)))
}

// WorkTree is the checkout as it is on disk: tracked files that still exist, plus
// untracked files git does not ignore.
func WorkTree(root string) (Source, error) {
	out, err := git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
	if err != nil {
		return nil, err
	}
	s := &workSource{root: root}
	for _, f := range splitZ(out) {
		if _, err := os.Lstat(filepath.Join(root, filepath.FromSlash(f))); err == nil {
			s.files = append(s.files, f)
		}
	}
	return s, nil
}

// Changed lists the paths that differ between ref and to; to == "" means the working
// tree, untracked files included. Renames are reported as a delete and an add, so each
// side is judged where it lives.
func Changed(root, ref, to string) ([]string, error) {
	args := []string{"diff", "-z", "--name-only", "--no-renames", ref}
	if to != "" {
		args = append(args, to)
	}
	out, err := git(root, args...)
	if err != nil {
		return nil, err
	}
	paths := splitZ(out)
	if to == "" {
		un, err := git(root, "ls-files", "-z", "--others", "--exclude-standard")
		if err != nil {
			return nil, err
		}
		paths = append(paths, splitZ(un)...)
	}
	return paths, nil
}

// Between is Compare for two refs in a checkout; to == "" means the working tree.
func Between(root, from, to string) (Mark, error) {
	old, err := AtRef(root, from)
	if err != nil {
		return Mark{}, err
	}
	var new Source
	if to == "" {
		new, err = WorkTree(root)
	} else {
		new, err = AtRef(root, to)
	}
	if err != nil {
		return Mark{}, err
	}
	changed, err := Changed(root, from, to)
	if err != nil {
		return Mark{}, err
	}
	return Compare(old, new, changed), nil
}

// RootTags lists the root module's release tags (vX.Y.Z), highest version first.
func RootTags(root string) ([]string, error) {
	out, err := git(root, "tag", "--list", "v*", "--sort=-v:refname")
	if err != nil {
		return nil, err
	}
	return strings.Fields(string(out)), nil
}
