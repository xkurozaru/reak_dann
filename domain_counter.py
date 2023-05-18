import os


def main():
    counters = {}
    data_dir = "/data2/eto/Dataset/tomato_combine/train/plant_team/00_HEAL"
    # data_dir = "/data2/eto/Dataset/tomato_combine/test/29_YeLC"
    datas = os.listdir(data_dir)
    for data in datas:
        domain = data[0:2]
        if domain not in counters:
            counters[domain] = 0
        counters[domain] += 1
    lists = sorted(counters.items())
    for k, v in lists:
        print(k, v)


if __name__ == "__main__":
    main()
