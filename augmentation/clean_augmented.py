import argparse
import glob
import os
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Move augmented leftovers (files matching '*_?<digit>.*') to an 'empty' folder")
    parser.add_argument("path", help="folder to clean up")
    opt = parser.parse_args()

    path = opt.path

    # Check whether the specified path exists or not
    path_empty = os.path.join(path, "../empty")
    if not os.path.exists(path_empty):
        os.makedirs(path_empty)

    txt_list = list(glob.iglob(rf"{path}/*_?[0-9].*"))
    total = len(txt_list)
    print(f"TOTAL FILE: {total}\nFROM: {path}")

    for txt_key in txt_list:
        f_path, f_name = os.path.split(txt_key)
        shutil.move(src=txt_key,
                    dst=os.path.join(f_path, "../empty", f_name))
        total -= 1
        print(f"LEFT EMPTY: {total}")


if __name__ == '__main__':
    main()
