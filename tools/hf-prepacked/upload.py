"""A checked save uploaded to the Hub as a private repo, its files checked there against the local ones (names and
sizes) and its card, then made public only with --public. The token is HF_TOKEN in this process's environment (the
caller reads it into the environment; it is never written to a file or printed).

    HF_TOKEN=... python upload.py SAVE_DIR glyd/NAME README.md LICENSE [--public]

NOTICE is written beside the weights: the files stored in Glyd's format (Apache-2.0 section 4(b)).
"""
import os
import shutil
import sys

from huggingface_hub import HfApi


def main(save, repo, card, license_file, public=False):
    src = open(card).read().split("base_model: ", 1)[1].split("\n", 1)[0]
    shutil.copyfile(card, os.path.join(save, "README.md"))
    shutil.copyfile(license_file, os.path.join(save, "LICENSE"))
    with open(os.path.join(save, "NOTICE"), "w") as f:
        f.write(f"This repository holds {src} (https://huggingface.co/{src}), its weights unchanged, bit for bit.\n"
                "Changed from the original's files: the weights are stored in Glyd's format (model*.safetensors, glyd.json).\n")
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=save, commit_message=f"{src}, stored with Glyd")
    local = {f: os.path.getsize(os.path.join(save, f)) for f in os.listdir(save) if os.path.isfile(os.path.join(save, f))}
    remote = {e.path: e.size for e in api.list_repo_tree(repo, recursive=True) if hasattr(e, "size") and e.path != ".gitattributes"}
    if local != remote:
        sys.exit(f"{repo}: the Hub's files differ from {save}'s: {sorted(set(local.items()) ^ set(remote.items()))[:5]}")
    if open(api.hf_hub_download(repo, "README.md")).read() != open(card).read():
        sys.exit(f"{repo}: README.md is not the card")
    print(f"{repo}: {len(local)} files, {sum(local.values()):,} bytes, checked (private)")
    if public:
        api.update_repo_settings(repo, private=False)
        print(f"{repo}: public")


if __name__ == "__main__":
    main(*sys.argv[1:5], public="--public" in sys.argv[5:])
