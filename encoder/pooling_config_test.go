package encoder

import (
	"os"
	"path/filepath"
	"testing"
)

// TestPoolingFromConfig reads the sentence-transformers 1_Pooling declaration:
// cls/mean map to the mode, an absent file falls back to the family default, and
// an unsupported mode (max/mean_sqrt_len) is a hard error rather than a silent
// mispool.
func TestPoolingFromConfig(t *testing.T) {
	write := func(t *testing.T, body string) string {
		dir := t.TempDir()
		pd := filepath.Join(dir, "1_Pooling")
		if err := os.MkdirAll(pd, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(pd, "config.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return dir
	}

	cases := []struct {
		name, body string
		want       pooling
		wantErr    bool
	}{
		{"cls", `{"pooling_mode_cls_token":true,"pooling_mode_mean_tokens":false}`, poolCLS, false},
		{"mean", `{"pooling_mode_cls_token":false,"pooling_mode_mean_tokens":true}`, poolMean, false},
		{"max unsupported", `{"pooling_mode_max_tokens":true}`, "", true},
		{"sqrt unsupported", `{"pooling_mode_mean_sqrt_len_tokens":true}`, "", true},
		{"none set → fallback", `{"pooling_mode_cls_token":false,"pooling_mode_mean_tokens":false}`, poolMean, false},
	}
	for _, c := range cases {
		got, err := poolingFromConfig(write(t, c.body), poolMean)
		if c.wantErr {
			if err == nil {
				t.Errorf("%s: expected error, got mode %q", c.name, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: unexpected error %v", c.name, err)
		} else if got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}

	// Absent 1_Pooling → the fallback (a bare BERT export).
	if got, err := poolingFromConfig(t.TempDir(), poolCLS); err != nil || got != poolCLS {
		t.Errorf("absent file: got %q err=%v, want cls fallback", got, err)
	}
}

// TestBERT_poolingReadFromFixture asserts the real all-MiniLM fixture loads with
// mean pooling read from its 1_Pooling/config.json (not the old hardcoded
// default) — the parity-discipline assertion for declared pooling.
func TestBERT_poolingReadFromFixture(t *testing.T) {
	const dir = "../testdata/minilm-model"
	if _, err := os.Stat(dir + "/model.safetensors"); err != nil {
		t.Skipf("no MiniLM model at %s", dir)
	}
	b, err := LoadBERT(dir)
	if err != nil {
		t.Fatal(err)
	}
	if b.pool != poolMean {
		t.Errorf("all-MiniLM pool = %q, want mean (from 1_Pooling/config.json)", b.pool)
	}
}
