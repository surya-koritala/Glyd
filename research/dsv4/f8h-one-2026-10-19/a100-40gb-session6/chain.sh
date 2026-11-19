#!/bin/bash
# Wait for the PC's two library builds (sm_89 with line info; sm_80 + sm_90a), check the digits of every variant on the 4080 under the GPU lock,
# copy the datacenter library here, then launch the A100 session (launch.sh) if every variant gives the known digits.
S=/private/tmp/claude-501/-Users-suryakoritala/d09db505-88bb-4450-a82f-95fd523dc0ee/scratchpad/a100
PC="ssh -o ConnectTimeout=20 -o BatchMode=yes glyd-pc"
until $PC 'grep -q LIB-DONE ~/glyd-scratch/f8one4/lib.log && grep -q LIB-DONE ~/glyd-scratch/dc2/lib.log' 2>/dev/null; do sleep 30; done
echo "$(date +%T) builds done"
$PC 'grep -E "error" ~/glyd-scratch/f8one4/lib.log ~/glyd-scratch/dc2/lib.log | head; ls -la ~/glyd-scratch/f8one4/lib ~/glyd-scratch/dc2/lib; grep -A3 "f8h_lin2_kernel" ~/glyd-scratch/dc2/lib.log | grep -E "Compiling|Used|spill" | head -24; grep -c LIB-DONE ~/glyd-scratch/dc2/lib.log' > $S/build-info.txt 2>&1
cat $S/build-info.txt
$PC 'cat > ~/glyd-scratch/f8one4/check.sh' <<'EOF'
#!/bin/bash
export GLYD_GPU_LIB=$HOME/glyd-scratch/f8one4/lib/libglyd_gpu_cuda13.so GLYD_F8_LAYOUT=h
E=$HOME/glyd-scratch/dc166/glyd-engine
cd $HOME/glyd-scratch/f8one4
for v in 1:3:0 2:3:0 2:3:1 1:3:1 0:3:1; do IFS=: read -r one c pl <<< "$v"; export GLYD_F8H_ONE=$one GLYD_F8H_CTAS=$c GLYD_F8H_PLAIN=$pl
  echo "== $v dsv4check: $($E dsv4check $HOME/glyd-scratch/dsv4/rand --tokens 24 2>&1 | grep -E '^decode|rror')"
  echo "== $v f8moecheck: $($E f8moecheck --tokens 33 --peaked 2>&1 | grep -E 'largest|rror' | tr '\n' ' ')"
  echo "== $v f8moecheck big: $($E f8moecheck --rows 2048 --cols 4096 --experts 16 --k 4 --tokens 33 --peaked 2>&1 | grep -E 'largest|one token|rror' | tr '\n' ' ')"
done
echo CHECK-DONE
EOF
$PC 'chmod +x ~/glyd-scratch/f8one4/check.sh; flock ~/glyd-scratch/gpu.lock ~/glyd-scratch/f8one4/check.sh' > $S/pc-check.txt 2>&1
echo "$(date +%T) PC checks:"; cat $S/pc-check.txt
scp -q glyd-pc:glyd-scratch/dc2/lib/libglyd_gpu_cuda13.so $S/dc/libglyd_gpu_cuda13.so
ls -la $S/dc
if [ "$(grep -c '11 of 24 steps' $S/pc-check.txt)" = 5 ] && [ "$(grep -c 'largest difference 0.489' $S/pc-check.txt)" = 5 ] && [ "$(grep -c 'largest difference 0.446' $S/pc-check.txt)" = 5 ] && ! grep -q rror $S/pc-check.txt; then
  echo "$(date +%T) digits match: launching"
  cd $S && bash launch.sh > launch.log 2>&1
  tail -5 launch.log
else
  echo "$(date +%T) PC checks differ: no launch"
fi
