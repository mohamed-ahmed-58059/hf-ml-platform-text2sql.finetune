from datasets import Dataset, get_dataset_split_names, load_dataset
from huggingface_hub import HfApi, hf_hub_download

from src.model.example import Example


def fetch_from_hf(repo_id, split, token=None):
    ds = load_dataset(repo_id, split=split, token=token)
    examples = [Example.from_row(r) for r in ds]
    db_path = hf_hub_download(repo_id, f"dbs/{split}.db", repo_type="dataset", token=token)
    return examples, db_path


def push_to_hf(contexts, repo_id, token, overwrite=False, private=False):
    api = HfApi(token=token)

    if overwrite and api.repo_exists(repo_id, repo_type="dataset"):
        print(f"overwrite: deleting {repo_id}")
        api.delete_repo(repo_id, repo_type="dataset")

    existing = set()
    if api.repo_exists(repo_id, repo_type="dataset"):
        try:
            existing = set(get_dataset_split_names(repo_id, token=token))
        except Exception:
            existing = set()

    for ctx in contexts:
        split = ctx.split.name
        if split in existing and not overwrite:
            print(f"[{split}] already on the hub — skipping (set OVERWRITE=True to replace)")
            continue
        rows = [e.to_row() for e in ctx.examples]
        Dataset.from_list(rows).push_to_hub(repo_id, split=split, token=token, private=private)
        api.upload_file(
            path_or_fileobj=str(ctx.split.db_path),
            path_in_repo=f"dbs/{split}.db",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"[{split}] pushed {len(rows)} rows + {split}.db")
