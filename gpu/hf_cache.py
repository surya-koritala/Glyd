"""A repo's files from the local Hugging Face cache, each by its name.

huggingface_hub's snapshot_download(local_files_only=True) refuses a cached snapshot that lacks any file its patterns match (IncompleteSnapshotError, hub 1.x), and
a job that downloaded one layer of a model (the config, the index and the shards that hold the layer) leaves exactly such a snapshot. The tools of this directory read
the few files they need and ask for each of them alone."""
import os


def cached(repo, name, cache_dir=None):
    """The path of file `name` of `repo` (a Hugging Face id, or a directory that holds the files) as the cache has it; FileNotFoundError, saying which file, where it is not."""
    if os.path.isdir(repo):
        path = os.path.join(repo, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{repo}: no {name}")
        return path
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo, name, local_files_only=True, cache_dir=cache_dir)
    except Exception as e:  # (LocalEntryNotFoundError, or the offline mode's own: one message for both)
        raise FileNotFoundError(f"{repo}: {name} is not in the Hugging Face cache ({type(e).__name__}); a job's download of a layer leaves the config, the index and the layer's shards") from e
