import glob, os
data_dirs = (
        # "./data/l_eye",
        # "./data/r_eye",
        # "./data/face",
        # "./data/face_aligned",
        # "./data/head_pos",
        # test
        "./test_data/l_eye",
        "./test_data/r_eye",
        "./test_data/face",
        "./test_data/face_aligned",
        "./test_data/head_pos",
    )

for dir in data_dirs:
    filelist = glob.glob(os.path.join(dir, "*.jpg"))
    for f in filelist:
        os.remove(f)
