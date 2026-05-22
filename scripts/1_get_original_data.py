import os
import shutil
import sys
import kagglehub


def main():
    dataset = "birdy654/cifake-real-and-ai-generated-synthetic-images"

    # Download latest version (kagglehub returns a path to a file or directory)
    path = kagglehub.dataset_download(dataset)
    print("Downloaded path:", path)

    # Compute project "data" directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    if not path:
        print("No path returned by kagglehub.dataset_download", file=sys.stderr)
        sys.exit(1)

    # If the downloader returned a directory, copy its contents into data/
    if os.path.isdir(path):
        for name in os.listdir(path):
            src = os.path.join(path, name)
            dest = os.path.join(data_dir, name)

            # If destination exists, remove it so we overwrite with fresh copy
            if os.path.exists(dest):
                if os.path.isdir(dest):
                    shutil.rmtree(dest)
                else:
                    os.remove(dest)

            if os.path.isdir(src):
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)

        print(f"Copied directory contents to {data_dir}")

    # If the downloader returned a file, try to extract common archive formats, otherwise copy
    elif os.path.isfile(path):
        try:
            shutil.unpack_archive(path, data_dir)
            print(f"Extracted archive to {data_dir}")
        except Exception:
            dest = os.path.join(data_dir, os.path.basename(path))
            shutil.copy2(path, dest)
            print(f"Copied file to {dest}")

    else:
        print(f"Downloaded path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    print("Data is available in:", data_dir)


if __name__ == "__main__":
    main()
