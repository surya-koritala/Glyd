#!/bin/bash
# transcript.sh: a first run as a person has it, in a terminal: the installer from nothing (empty uv cache), then `glyd run Qwen/Qwen3-8B` with an empty
# model cache on a 16 GB card's memory (the L4 with a process holding the rest), a question typed at the chat prompt and /bye. The terminal's bytes in
# ~/accept-onb/transcript/logs/{install,run}.typescript. All of it under the GPU lock (the doctor at the end of the install wants an idle GPU).
set -u
[ -n "${INLOCK:-}" ] || exec env INLOCK=1 flock "$HOME/.glyd-box.lock" bash "$0"
touch ~/.glyd-busy; trap 'rm -f ~/.glyd-busy' EXIT
echo "== started $(date -u +%T)"
git -C ~/onb/src pull -q
W=$HOME/accept-onb/transcript
SRC=$HOME/onb/src
WHEEL=$(ls $HOME/onb/wheels/*.whl | head -1)
IMAGE=glyd-accept:ubuntu26.04-curl
rm -rf $W; mkdir -p $W/home $W/hf $W/uv-cache $W/logs $W/wheel
cp $SRC/scripts/install.sh $W/; cp $WHEEL $W/wheel/; cp $HOME/onb/ptydrive.py $W/
cp -r $HOME/quick/geforce $HOME/quick/hog.py $W/
NAME=glyd-transcript
docker rm -f $NAME > /dev/null 2>&1
docker run -d --rm --name $NAME --gpus all --network host --ipc host --user "$(id -u):$(id -g)" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HOME=/work/home -e UV_CACHE_DIR=/work/uv-cache -e HF_HOME=/work/hf -e PATH=/work/home/.local/bin:/usr/local/bin:/usr/bin:/bin -e TERM=xterm-256color \
  -v $W:/work $IMAGE sleep infinity > /dev/null || exit 1
t0=$(date +%s)
docker exec -i -e GLYD_SPEC="/work/wheel/$(basename $WHEEL)[vllm]" $NAME script -qec "stty cols 100 rows 30; sh /work/install.sh" /work/logs/install.typescript > /dev/null
echo "install: $(( $(date +%s) - t0 )) s; tool dir $(docker exec $NAME du -sh /work/home/.local/share/uv/tools/glyd | cut -f1), python $(docker exec $NAME du -sh /work/home/.local/share/uv/python | cut -f1), uv cache $(docker exec $NAME du -sh /work/uv-cache | cut -f1), all of it on disk: $(du -sh $W | cut -f1)"
TOOLPY=/work/home/.local/share/uv/tools/glyd/bin/python
docker exec -d $NAME bash -c "$TOOLPY /work/hog.py $((14828 + 190)) > /work/logs/hog.log 2>&1"
for i in $(seq 1 60); do grep -q '^hog:' $W/logs/hog.log 2>/dev/null && break; sleep 1; done; cat $W/logs/hog.log
t1=$(date +%s)
docker exec -i -e PYTHONPATH=/work/geforce $NAME $TOOLPY /work/ptydrive.py /work/logs/run.typescript '>>> =Say hello in five words.' '>>> =/bye' -- glyd run Qwen/Qwen3-8B
echo "run: rc=$? after $(( $(date +%s) - t1 )) s"
du -sh $W/hf | cut -f1 | sed 's/^/model cache: /'
docker rm -f $NAME > /dev/null 2>&1
echo done
