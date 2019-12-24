import shutil
from unidecode import unidecode
from pathlib import Path
import stat


def file_is_hidden(file_path: Path):
    return file_path.stat().st_file_attributes & (stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM)


def correct_path_character(c: str):
    return c.isalnum() or c == '\\' or c == ' ' or c == '.' or c == '_'


def create_dir_copy_with_latin_file_names(dir_path: Path):
    result_dir_path = dir_path.parent / (dir_path.name + '_latin')
    result_dir_path.mkdir()
    for item_path in dir_path.rglob('*'):
        if file_is_hidden(item_path):
            continue

        item_relative_path = item_path.relative_to(dir_path)
        # Convert to latin characters
        item_latin_relative_path = unidecode(str(item_relative_path))
        item_latin_relative_path = ''.join(filter(correct_path_character, item_latin_relative_path))

        dst_item_path = result_dir_path / item_latin_relative_path
        if item_path.is_dir():
            dst_item_path.mkdir()
        else:
            shutil.copyfile(item_path, dst_item_path)

    return result_dir_path
