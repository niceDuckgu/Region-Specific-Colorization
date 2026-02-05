from huggingface_hub import HfApi, create_repo
import os
import argparse

def upload_checkpoints(repo_id, folder_path, token=None):
    print(f"Uploading checkpoints from {folder_path} to Hugging Face Hub: {repo_id}")
    
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", token=token, exist_ok=True)
    except Exception as e:
        print(f"Repo creation warning (might already exist): {e}")

    # Upload folder
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="checkpoints",
        ignore_patterns=[".git", "__pycache__"]
    )
    print("Upload complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID (e.g., username/model-name)")
    parser.add_argument("--ckpt_dir", type=str, default="src/ckpts", help="Local directory containing checkpoints")
    parser.add_argument("--token", type=str, help="Hugging Face User Access Token (write permission)")
    args = parser.parse_args()
    
    upload_checkpoints(args.repo_id, args.ckpt_dir, args.token)

