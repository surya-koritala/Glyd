"""The Hugging Face workflow's steps (ACTION: whoami or card). Reads HF_TOKEN from the environment; never prints it."""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main():
    if not os.environ.get("HF_TOKEN"):
        sys.exit("no HUGGINGFACE secret in this environment")
    api = HfApi()
    me = api.whoami()
    orgs = {o["name"]: o.get("roleInOrg") for o in me.get("orgs", [])}
    action = os.environ.get("ACTION", "whoami")
    if action == "whoami":
        print(f"account: {me['name']} ({me.get('type')})")
        print("organizations: " + (", ".join(f"{n} ({r})" for n, r in sorted(orgs.items())) or "none"))
        return
    org = os.environ.get("ORG", "").strip()
    if action != "card" or not org:
        sys.exit("card needs an organization")
    if orgs.get(org) not in ("admin", "write", "contributor"):
        sys.exit(f"the token's account cannot write to {org} (role: {orgs.get(org)})")
    repo = f"{org}/README"  # an organization's card is the README of its Space named README
    api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    card = Path(__file__).with_name("org-card.md")
    info = api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo, repo_type="space",
                           commit_message="Organization card")
    print(f"published {card.name} to https://huggingface.co/{org} ({info.oid})")


if __name__ == "__main__":
    main()
