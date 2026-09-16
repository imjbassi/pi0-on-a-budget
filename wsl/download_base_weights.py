"""Download pi0_fast_base weights into $OPENPI_DATA_HOME and report their size."""
import subprocess

from openpi.shared import download

path = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_base/params")
print("weights at:", path)
print(subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout)
