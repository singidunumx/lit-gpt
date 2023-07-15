import os
from typing import Optional

from download import download_from_hub
from convert_hf_checkpoint import convert_hf_checkpoint


def download_and_convert(
    checkpoint: Optional[str] = None,
    checkpoint_dir: Optional[str] = None
):
    download_from_hub(repo_id=checkpoint)

    if checkpoint:
        return

    if checkpoint_dir is None:
        checkpoint_dir = "checkpoints"

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    convert_hf_checkpoint(checkpoint_path)

    print(f"Model checkpoint ready at: {checkpoint_path}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_and_convert)
