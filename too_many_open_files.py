files = []
for i in range(10000):
    with open("files/some_file.txt", "r") as f:
        files.append(f.read())
    print(i)