package main

import (
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/tools/gate"
	"github.com/townsendmerino/aikit/tools/retrievalzone"
)

func TestJudgeRetrievalMark(t *testing.T) {
	yes := retrievalzone.Mark{Yes: true, Files: []retrievalzone.File{{Path: "ann/flat.go", Zone: true, Why: "in zone package ann"}, {Path: "README.md"}}}
	no := retrievalzone.Mark{Files: []retrievalzone.File{{Path: "vision/tower.go"}}}
	for _, tc := range []struct {
		name string
		sec  string
		mark retrievalzone.Mark
		want gate.Outcome
		msg  string
	}{
		{"the tool's own line, no", "\n" + no.Line("v1.0.0") + "\n\n### Added\n", no, gate.OK, ""},
		{"the tool's own line, yes", "\n" + yes.Line("v1.0.0") + "\n", yes, gate.OK, ""},
		{"no line", "\n### Added\n- a thing\n", no, gate.Fail, "has no RETRIEVAL line"},
		// The failure the check exists for.
		{"says no, files say yes", "\nRETRIEVAL: no — vs v1.0.0, nothing\n", yes, gate.Fail, "says RETRIEVAL: no, but 1 file(s)"},
		// The author may say yes over a computed no.
		{"says yes, files say no", "\nRETRIEVAL: yes — a dependency of ken's changed behavior\n", no, gate.OK, ""},
		// Prose that mentions the word is not the mark.
		{"prose only", "\nThis release has no RETRIEVAL: no line yet.\n", no, gate.Fail, "has no RETRIEVAL line"},
		{"indented or quoted", "\n> RETRIEVAL: no — quoted\n", no, gate.Fail, "has no RETRIEVAL line"},
		{"another word", "\nRETRIEVAL: nope\n", no, gate.Fail, "has no RETRIEVAL line"},
		{"two lines", "\nRETRIEVAL: no — a\nRETRIEVAL: yes — b\n", yes, gate.Fail, "2 RETRIEVAL lines"},
	} {
		c := judgeRetrievalMark(tc.sec, "1.1.0", "v1.0.0", tc.mark)
		if c.Outcome != tc.want || !strings.Contains(c.Field("msg"), tc.msg) {
			t.Errorf("%s: outcome %s, msg %q; want %s, msg containing %q", tc.name, c.Outcome, c.Field("msg"), tc.want, tc.msg)
		}
	}
}
