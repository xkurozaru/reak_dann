import os
from random import sample
from shutil import copy, move


def leak():
    """TestからLeakする"""
    test_root = "/data2/eto/Dataset/eggplant_fewclass/7class_leak/test/"
    leak_root = "/data2/eto/Dataset/eggplant_fewclass/7class_leak/leak/"
    dirlist = os.listdir(test_root)
    dirlist.sort()

    for dir in dirlist:
        print(f"===========\n{dir}\n===========")
        if not os.path.exists(os.path.join(leak_root, dir)):
            os.mkdir(os.path.join(leak_root, dir))

        files = os.listdir(os.path.join(test_root, dir))
        num_leaks = int(len(files) * 0.01)
        leaks = sample(files, num_leaks)

        for leak in leaks:
            move(os.path.join(test_root, dir, leak), os.path.join(leak_root, dir))
            print(os.path.join(leak_root, dir, leak))


def restore():
    """Leakから戻す"""
    test_root = "/data2/eto/Dataset/eggplant_fewclass/7class_leak/test/"
    leak_root = "/data2/eto/Dataset/eggplant_fewclass/7class_leak/leak/"
    dirlist = os.listdir(leak_root)
    dirlist.sort()

    for dir in dirlist:
        print(f"===========\n{dir}\n===========")
        files = os.listdir(os.path.join(leak_root, dir))

        for file in files:
            move(os.path.join(leak_root, dir, file), os.path.join(test_root, dir))
            print(os.path.join(test_root, dir, file))


def remake():
    """データセットをコピーする"""
    from_dir = ""
    to_dir = ""
    dirlist = os.listdir(from_dir)
    dirlist.sort()

    for dir in dirlist:
        print(f"===========\n{dir}\n===========")
        files = os.listdir(os.path.join(from_dir, dir))
        if not os.path.exists(os.path.join(to_dir, dir)):
            os.mkdir(os.path.join(to_dir, dir))
        for file in files:
            copy(os.path.join(from_dir, dir, file), os.path.join(to_dir, dir))


if __name__ == "__main__":
    # restore()
    leak()
    # remake()
