# A user's first install and runs of v0.29.3: no uv, no glyd, an empty uv cache and an empty Hugging Face cache; the install script
# of the v0.29.3 tag (the one getglyd.com/install.sh serves once the site is updated).
F=/root/fresh3; R=$F/results.txt; mkdir -p $F; : > $R
export PATH=$HOME/.local/bin:$PATH
uv tool uninstall glyd > /dev/null 2>&1; rm -f ~/.local/bin/glyd ~/.local/bin/uv ~/.local/bin/uvx
export UV_CACHE_DIR=$F/uv HF_HOME=$F/hf
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >> $R
s=$(date +%s); curl -LsSf https://raw.githubusercontent.com/surya-koritala/Glyd/v0.29.3/scripts/install.sh | sh > $F/install.log 2>&1; echo "install: rc=$? $(( $(date +%s) - s )) s" >> $R
echo "glyd: $(glyd --version 2>&1 | head -1); glyd-gpu: $(~/.local/share/uv/tools/glyd/bin/python -c 'import glyd_gpu; print(glyd_gpu.__version__)' 2>&1)" >> $R
s=$(date +%s); glyd run Qwen/Qwen3.8-27B --lossy -p "In one sentence, why is the sky blue?" > $F/run1.out 2> $F/run1.err; echo "first run (download, pack, answer): rc=$? $(( $(date +%s) - s )) s" >> $R
s=$(date +%s); glyd run Qwen/Qwen3.8-27B --lossy --no-think -p "Name three primes above 100." > $F/run2.out 2> $F/run2.err; echo "second run: rc=$? $(( $(date +%s) - s )) s" >> $R
echo done >> $R
