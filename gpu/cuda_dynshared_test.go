//go:build linux

package gpu

import "testing"

// dynSharedPTX's touch_dyn writes 0x1234ABCD at byte offset off of its dynamic shared memory, reads it back and stores it to out[0].
const dynSharedPTX = `
.version 7.0
.target sm_75
.address_size 64
.extern .shared .align 4 .b8 dyn[];
.visible .entry touch_dyn(.param .u64 out, .param .u32 off)
{
	.reg .b32 %r<5>;
	.reg .b64 %rd<3>;
	ld.param.u64 %rd1, [out];
	ld.param.u32 %r1, [off];
	mov.u32 %r2, dyn;
	add.u32 %r3, %r2, %r1;
	mov.u32 %r4, 305441741;
	st.shared.u32 [%r3], %r4;
	ld.shared.u32 %r4, [%r3];
	cvta.to.global.u64 %rd2, %rd1;
	st.global.u32 [%rd2], %r4;
	ret;
}
`

// TestCUDA_maxDynamicSharedBytes: a launch asking for more than the 48 KiB default fails until the kernel opts in with
// SetMaxDynamicSharedBytes, then runs and reads back the word it wrote at the far end of that shared memory; a request past
// the device's opt-in maximum is refused.
func TestCUDA_maxDynamicSharedBytes(t *testing.T) {
	d, err := CreateSystemDefaultDevice()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	t.Cleanup(d.ReleaseObjects)
	opt := d.MaxSharedBytesPerBlockOptIn()
	if opt <= 48*1024 {
		t.Skipf("this device's opt-in maximum is %d bytes: no room past the 48 KiB default", opt)
	}
	lib, err := d.CompileLibrary([]byte(dynSharedPTX))
	if err != nil {
		t.Fatalf("CompileLibrary: %v", err)
	}
	p, err := d.NewComputePipeline(lib, "touch_dyn")
	if err != nil {
		t.Fatal(err)
	}
	q := d.NewCommandQueue()
	out := NewBufferOf(d, []uint32{0})
	run := func(shared int) error {
		if err := q.Launch(p, GridOne(1, shared), Arg(out), ArgValue(uint32(shared-4))); err != nil {
			return err
		}
		return q.Sync()
	}
	if err := run(48 * 1024); err != nil {
		t.Fatalf("a 48 KiB launch failed without any opt-in: %v", err)
	}
	if err := run(opt); err == nil {
		t.Fatalf("a %d-byte launch ran without the opt-in: the test cannot see the attribute", opt)
	}
	if err := p.SetMaxDynamicSharedBytes(opt); err != nil {
		t.Fatalf("SetMaxDynamicSharedBytes(%d): %v", opt, err)
	}
	if err := run(opt); err != nil {
		t.Fatalf("a %d-byte launch after the opt-in: %v", opt, err)
	}
	got := make([]uint32, 1)
	if err := Download(out, got); err != nil {
		t.Fatal(err)
	}
	if got[0] != 0x1234ABCD {
		t.Fatalf("read back %#x from the last word of %d bytes of shared memory, want 0x1234abcd", got[0], opt)
	}
	if err := p.SetMaxDynamicSharedBytes(opt + 1024); err == nil {
		t.Errorf("SetMaxDynamicSharedBytes(%d) past the opt-in maximum %d was accepted", opt+1024, opt)
	}
	t.Logf("opt-in maximum %d bytes; a launch at it runs only after the opt-in", opt)
}
