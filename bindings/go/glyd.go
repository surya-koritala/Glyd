// Package glyd: compression for the data that fills object storage.
//
//	c, _ := glyd.Compress(data, glyd.LevelMax, false)   // records=true for logs, dumps, CSV, JSON lines
//	data, _ := glyd.Decompress(c)                        // any Glyd stream
//	p, _ := glyd.Pack(objects, glyd.LevelMax)            // many small objects as one stream
//	obj, _ := glyd.Unpack(p, 7)
//	s, _ := glyd.OpenStore("bucket/", "")                // objects compressed across each other
//	id, _ := s.Put("wed.tar", data)                      // a delta against the object it most resembles
//	data, _ = s.Get(id)
//	s.Close()
//
// Binds include/glyd.h through cgo; build the libraries first (cargo
// build --release --workspace) and point cgo at them, as the flags
// below do for a checkout. libglyd_store carries the codec and the
// store (the store is BUSL-1.1); link -lglyd instead for the codec
// alone (BSD-3-Clause OR GPL-2.0), dropping Store.
package glyd

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../target/release -lglyd_store
#cgo darwin LDFLAGS: -framework CoreFoundation
#include <stdlib.h>
#include "glyd.h"
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Levels.
const (
	LevelDefault = C.GLYD_LEVEL_DEFAULT
	LevelFast    = C.GLYD_LEVEL_FAST
	LevelTurbo   = C.GLYD_LEVEL_TURBO
	LevelMax     = C.GLYD_LEVEL_MAX
	LevelUltra   = C.GLYD_LEVEL_ULTRA
	LevelCold    = C.GLYD_LEVEL_COLD
	LevelMaxLong = C.GLYD_LEVEL_MAX_LONG
)

var (
	ErrArgument = errors.New("glyd: bad argument")
	ErrCorrupt  = errors.New("glyd: not a Glyd stream, corrupt, or the wrong base")
	ErrStore    = errors.New("glyd: store operation failed")
)

// Version of the library.
func Version() string { return C.GoString(C.glyd_version()) }

func ptr(b []byte) *C.uint8_t {
	if len(b) == 0 {
		return nil
	}
	return (*C.uint8_t)(unsafe.Pointer(&b[0]))
}

// take copies a library buffer into Go memory and frees it.
func take(out *C.uint8_t, n C.size_t) []byte {
	b := C.GoBytes(unsafe.Pointer(out), C.int(n))
	C.glyd_free(out, n)
	return b
}

func status(r C.int) error {
	switch r {
	case 0:
		return nil
	case -1:
		return ErrArgument
	default:
		return ErrCorrupt
	}
}

// Compress at a level, in record mode when records is true (logs,
// dumps, CSV, JSON lines as typed columns); threads 0 uses every core.
func Compress(data []byte, level int, records bool) ([]byte, error) {
	return CompressThreads(data, level, records, 0)
}

func CompressThreads(data []byte, level int, records bool, threads int) ([]byte, error) {
	var out *C.uint8_t
	var n C.size_t
	rec := 0
	if records {
		rec = 1
	}
	if err := status(C.glyd_compress2(ptr(data), C.size_t(len(data)), C.int(level), C.int(rec), C.int(threads), &out, &n)); err != nil {
		return nil, err
	}
	return take(out, n), nil
}

// Decompress any Glyd stream (a base-mode stream needs DecompressWithBase).
func Decompress(data []byte) ([]byte, error) {
	var out *C.uint8_t
	var n C.size_t
	if err := status(C.glyd_decompress2(ptr(data), C.size_t(len(data)), &out, &n)); err != nil {
		return nil, err
	}
	return take(out, n), nil
}

// DecompressedLen of a stream.
func DecompressedLen(data []byte) (int64, error) {
	r := C.glyd_decompressed_len(ptr(data), C.size_t(len(data)))
	if r < 0 {
		return 0, ErrCorrupt
	}
	return int64(r), nil
}

// CompressWithBase: a new version against an old one; decoding needs the same base.
func CompressWithBase(base, data []byte, ultra bool) ([]byte, error) {
	var out *C.uint8_t
	var n C.size_t
	u := 0
	if ultra {
		u = 1
	}
	if err := status(C.glyd_compress_with_base(ptr(base), C.size_t(len(base)), ptr(data), C.size_t(len(data)), C.int(u), &out, &n)); err != nil {
		return nil, err
	}
	return take(out, n), nil
}

func DecompressWithBase(base, data []byte) ([]byte, error) {
	var out *C.uint8_t
	var n C.size_t
	if err := status(C.glyd_decompress_with_base(ptr(base), C.size_t(len(base)), ptr(data), C.size_t(len(data)), &out, &n)); err != nil {
		return nil, err
	}
	return take(out, n), nil
}

// Pack many small objects as one stream with an index (level Max, Ultra or Cold).
func Pack(objects [][]byte, level int) ([]byte, error) {
	// The objects and the pointer table go through C memory: cgo does
	// not allow a table of Go pointers to cross.
	k := len(objects)
	total := 0
	for _, o := range objects {
		total += len(o)
	}
	buf := (*C.uint8_t)(C.malloc(C.size_t(total + 1)))
	ptrs := (**C.uint8_t)(C.malloc(C.size_t((k + 1) * int(unsafe.Sizeof(uintptr(0))))))
	lens := (*C.size_t)(C.malloc(C.size_t((k + 1) * int(unsafe.Sizeof(C.size_t(0))))))
	defer C.free(unsafe.Pointer(buf))
	defer C.free(unsafe.Pointer(ptrs))
	defer C.free(unsafe.Pointer(lens))
	pt := unsafe.Slice(ptrs, k+1)
	ln := unsafe.Slice(lens, k+1)
	at := 0
	all := unsafe.Slice((*byte)(unsafe.Pointer(buf)), total+1)
	for i, o := range objects {
		copy(all[at:], o)
		pt[i] = (*C.uint8_t)(unsafe.Add(unsafe.Pointer(buf), at))
		ln[i] = C.size_t(len(o))
		at += len(o)
	}
	var out *C.uint8_t
	var n C.size_t
	if err := status(C.glyd_pack(ptrs, lens, C.size_t(k), C.int(level), &out, &n)); err != nil {
		return nil, err
	}
	return take(out, n), nil
}

// Unpack object i of a pack.
func Unpack(pack []byte, i int) ([]byte, error) {
	var out *C.uint8_t
	var n C.size_t
	if err := status(C.glyd_unpack_object(ptr(pack), C.size_t(len(pack)), C.size_t(i), &out, &n)); err != nil {
		return nil, err
	}
	return take(out, n), nil
}

// PackLen: the object count of a pack.
func PackLen(pack []byte) (int, error) {
	r := C.glyd_pack_len(ptr(pack), C.size_t(len(pack)))
	if r < 0 {
		return 0, ErrCorrupt
	}
	return int(r), nil
}

// Store: objects compressed across each other. Metadata at dir; the
// objects there too, or in s3 (s3://bucket/prefix)
// when it is not empty.
type Store struct{ h *C.GlydStore }

func OpenStore(dir, s3 string) (*Store, error) {
	cdir := C.CString(dir)
	defer C.free(unsafe.Pointer(cdir))
	var cs3 *C.char
	if s3 != "" {
		cs3 = C.CString(s3)
		defer C.free(unsafe.Pointer(cs3))
	}
	h := C.glyd_store_open(cdir, cs3)
	if h == nil {
		return nil, ErrStore
	}
	return &Store{h: h}, nil
}

// Close writes the open pack and releases the store.
func (s *Store) Close() {
	if s.h != nil {
		C.glyd_store_close(s.h)
		s.h = nil
	}
}

// Put stores data under name; its id.
func (s *Store) Put(name string, data []byte) (uint32, error) {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	r := C.glyd_store_put(s.h, cname, ptr(data), C.size_t(len(data)))
	if r < 0 {
		return 0, ErrStore
	}
	return uint32(r), nil
}

func (s *Store) Get(id uint32) ([]byte, error) {
	var out *C.uint8_t
	var n C.size_t
	if r := C.glyd_store_get(s.h, C.uint32_t(id), &out, &n); r != 0 {
		return nil, ErrStore
	}
	return take(out, n), nil
}

// IDOf: the latest live object under name, or -1.
func (s *Store) IDOf(name string) int64 {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	return int64(C.glyd_store_id_of(s.h, cname))
}

func (s *Store) Delete(id uint32) error {
	if C.glyd_store_delete(s.h, C.uint32_t(id)) != 0 {
		return ErrStore
	}
	return nil
}

// Compact frees what no live object needs; the bytes freed.
func (s *Store) Compact() int64 { return int64(C.glyd_store_compact(s.h)) }
func (s *Store) Flush() error {
	if C.glyd_store_flush(s.h) != 0 {
		return ErrStore
	}
	return nil
}
func (s *Store) Rebase(id uint32) error {
	if C.glyd_store_rebase(s.h, C.uint32_t(id)) != 0 {
		return ErrStore
	}
	return nil
}

// Verify reads every live object back; the number that failed.
func (s *Store) Verify() int64 { return int64(C.glyd_store_verify(s.h)) }

// Stats: raw bytes of the live objects and bytes on disk.
func (s *Store) Stats() (raw, stored uint64) {
	var r, st C.uint64_t
	C.glyd_store_stats(s.h, &r, &st)
	return uint64(r), uint64(st)
}

func (s *Store) SetLevel(level int) error {
	if C.glyd_store_set_level(s.h, C.int(level)) != 0 {
		return ErrArgument
	}
	return nil
}

// Count of objects the store knows (ids run 0..Count).
func (s *Store) Count() int { return int(C.glyd_store_count(s.h)) }
