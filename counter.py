import os


def main():
    sum = 0
    print("----------------")
    data_dir = "/data2/eto/New_Plant_Diseases_Dataset_Augmented/valid/"
    folders = os.listdir(data_dir)
    folders.sort()
    for folder in folders:
        folder_dir = data_dir + folder
        print(f"{folder}: {len(os.listdir(folder_dir))}")
        sum += len(os.listdir(folder_dir))
    print(f"sum: {sum}")
    print("---------------------")


if __name__ == "__main__":
    main()
