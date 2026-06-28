#!/usr/bin/env python3
import os
import argparse
from huggingface_hub import HfApi, login

def upload_to_hub(repo_id, local_path, path_in_repo=None):
    """
    Belirtilen yerel dosya veya klasörü Hugging Face Hub reposuna yükler.
    """
    print(f"🔄 Hugging Face Hub'a yükleniyor: {local_path} -> {repo_id}")
    api = HfApi()
    
    if os.path.isdir(local_path):
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            path_in_repo=path_in_repo
        )
    else:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo or os.path.basename(local_path),
            repo_id=repo_id
        )
    print("✅ Yükleme başarıyla tamamlandı!")

def download_from_hub(repo_id, filename, local_dir):
    """
    Hugging Face Hub'dan model ağırlığı veya tokenizer dosyası indirir.
    """
    print(f"🔄 İndiriliyor: {filename} from {repo_id} -> {local_dir}")
    from huggingface_hub import hf_hub_download
    
    os.makedirs(local_dir, exist_ok=True)
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir
    )
    print(f"✅ İndirildi: {file_path}")
    return file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DarkMind Hugging Face Hub Sync Utility")
    parser.add_argument("--action", choices=["upload", "download"], required=True, help="Yapılacak eylem")
    parser.add_argument("--repo_id", required=True, help="Hugging Face repo kimliği (örn: petrofi/darkmind-30m)")
    parser.add_argument("--path", required=True, help="Yerel dosya/klasör yolu veya indirilecek dosya adı")
    parser.add_argument("--dest", help="Hedef yerel dizin veya repodaki hedef yol")
    parser.add_argument("--token", help="HF Write Token (Yükleme için gerekebilir)")

    args = parser.parse_args()

    if args.token:
        login(token=args.token)

    if args.action == "upload":
        upload_to_hub(args.repo_id, args.path, args.dest)
    elif args.action == "download":
        download_from_hub(args.repo_id, args.path, args.dest or "./checkpoints")
