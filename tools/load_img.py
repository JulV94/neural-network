def load_img(img_path):
    with open(img_path) as f:
        data = [x for x in f if not x.startswith('#')]  # remove comments
    p = data.pop(0)  # P thing
    dim = tuple(map(int, data.pop(0).split()))
    arr = []
    for line in data:
        for c in line.strip():
            arr.append(int(c))
    return dim, arr
