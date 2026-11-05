"""
Python-UNO Bridge for headless LibreOffice Impress control.
Enables programmatic connection to local headless LibreOffice daemons
via UNIX socket or TCP port 2002 using system python3-uno.
"""

from __future__ import annotations
import os
import subprocess
import sys
from typing import Optional


def is_system_uno_available() -> bool:
    """Checks if /usr/bin/python3 has access to the uno module."""
    try:
        res = subprocess.run(
            ["/usr/bin/python3", "-c", "import uno; print('OK')"],
            capture_output=True,
            text=True,
        )
        return res.returncode == 0 and "OK" in res.stdout
    except Exception:
        return False


def convert_presentation_uno(
    input_file: str,
    output_format: str = "pdf",
    output_dir: Optional[str] = None,
) -> str:
    """
    Executes headless LibreOffice presentation conversion using either direct CLI
    or UNO daemon bridge.
    """
    out_dir = output_dir or os.path.dirname(os.path.abspath(input_file))
    abs_input = os.path.abspath(input_file)
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        output_format,
        abs_input,
        "--outdir",
        out_dir,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"LibreOffice conversion failed: {res.stderr}")

    base_name = os.path.splitext(os.path.basename(abs_input))[0]
    output_file = os.path.join(out_dir, f"{base_name}.{output_format}")
    return output_file


if __name__ == "__main__":
    print(f"System UNO available: {is_system_uno_available()}")
    if len(sys.argv) > 1:
        target = sys.argv[1]
        pdf = convert_presentation_uno(target)
        print(f"Converted to PDF: {pdf}")
