"""用文件白名单生成可在 AstrBot WebUI 上传的安装包。"""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PLUGIN_NAME = "astrbot_plugin_keyword_wakeup"
VERSION = "v1.0.2"
PACKAGE_FILES = ("main.py", "_conf_schema.json", "metadata.yaml", "README.md", "LICENSE", "docs/DEVELOPMENT.md")


def build_package(output_dir: Path | None = None) -> Path:
    root = Path(__file__).resolve().parent
    plugin_dir = root
    destination = (output_dir or root.parent) / f"{PLUGIN_NAME}_{VERSION}.zip"
    # 在创建 ZIP 前检查源文件，缺少文件时不会覆盖已有安装包。
    for name in PACKAGE_FILES:
        if not (plugin_dir / name).is_file():
            raise FileNotFoundError(plugin_dir / name)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for name in PACKAGE_FILES:
            archive.write(plugin_dir / name, f"{PLUGIN_NAME}/{name}")
    return destination


if __name__ == "__main__":
    print(build_package())
