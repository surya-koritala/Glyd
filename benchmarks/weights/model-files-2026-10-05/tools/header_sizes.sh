#!/bin/sh
# header-only sizes: zstd -19 and glyd --max (v0.28.0) on the bytes before the tensor data of one corpus file
# run in the directory that holds corpus/ (with MANIFEST.tsv) and bin/glyd-base (the v0.28.0 glyd)
f=$1
name=$(basename $f)
hdr=$(awk -F'\t' -v n=$name '$2==n {print $4}' corpus/MANIFEST.tsv)
head -c $hdr $f > /tmp/codecw_h_$name
z=$(zstd -19 -T1 -q -c /tmp/codecw_h_$name | wc -c)
g=$(bin/glyd-base --max -s /tmp/codecw_h_$name -o /dev/stdout | wc -c)
x=$(xz -9 -T1 -c /tmp/codecw_h_$name | wc -c)
printf "%s\t%s\t%s\t%s\t%s\n" $name $hdr $z $g $x
rm -f /tmp/codecw_h_$name
