package glyd

import (
	"bytes"
	"fmt"
	"os"
	"testing"
)

func TestRoundTrips(t *testing.T) {
	var buf bytes.Buffer
	for i := 0; i < 20000; i++ {
		fmt.Fprintf(&buf, `{"ts": %d, "host": "h%d", "n": %d}`+"\n", 1700000000+i, i%20, (i*7919)%1000)
	}
	data := buf.Bytes()
	for _, level := range []int{LevelDefault, LevelFast, LevelMax, LevelUltra} {
		c, err := Compress(data, level, false)
		if err != nil {
			t.Fatal(err)
		}
		d, err := Decompress(c)
		if err != nil || !bytes.Equal(d, data) {
			t.Fatalf("level %d round trip", level)
		}
	}
	rec, _ := Compress(data, LevelMax, true)
	plain, _ := Compress(data, LevelMax, false)
	if d, _ := Decompress(rec); !bytes.Equal(d, data) || len(rec) >= len(plain) {
		t.Fatal("record mode")
	}
	if n, _ := DecompressedLen(rec); n != int64(len(data)) {
		t.Fatal("decompressed len")
	}
	v2 := append(append([]byte{}, data[:5000]...), append([]byte("changed\n"), data[5000:]...)...)
	delta, _ := CompressWithBase(data, v2, false)
	if d, _ := DecompressWithBase(data, delta); !bytes.Equal(d, v2) {
		t.Fatal("base mode")
	}
	var objs [][]byte
	for i := 0; i < 50; i++ {
		objs = append(objs, data[i*1000:(i+1)*1000])
	}
	p, _ := Pack(objs, LevelMax)
	if n, _ := PackLen(p); n != 50 {
		t.Fatal("pack len")
	}
	if o, _ := Unpack(p, 7); !bytes.Equal(o, objs[7]) {
		t.Fatal("unpack")
	}
	dir, _ := os.MkdirTemp("", "glyd-go")
	defer os.RemoveAll(dir)
	s, err := OpenStore(dir, "")
	if err != nil {
		t.Fatal(err)
	}
	i, _ := s.Put("v1", data)
	j, _ := s.Put("v2", v2)
	k, _ := s.Put("small", []byte("tiny"))
	if d, _ := s.Get(j); !bytes.Equal(d, v2) {
		t.Fatal("store get")
	}
	if d, _ := s.Get(k); string(d) != "tiny" {
		t.Fatal("store small")
	}
	if s.IDOf("v1") != int64(i) || s.Verify() != 0 {
		t.Fatal("store id/verify")
	}
	raw, stored := s.Stats()
	if stored*4 >= raw {
		t.Fatalf("store should shrink: %d of %d", stored, raw)
	}
	s.Close()
	s, _ = OpenStore(dir, "")
	if d, _ := s.Get(j); !bytes.Equal(d, v2) {
		t.Fatal("reopen")
	}
	s.Close()
	t.Logf("glyd %s go binding ok", Version())
}
