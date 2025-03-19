import os
target_path = "/mnt/sdb/dataset/ActivityNet_QA_test/video_frames"
output_file = "empty_folders_act_test.txt"

empty_folders = []
for root, dirs, files in os.walk(target_path):
    for folder in dirs:
        folder_path = os.path.join(root, folder)
        if not os.listdir(folder_path):
            empty_folders.append(folder_path.split('/')[-1])

with open(output_file, "w") as f:
    for folder in empty_folders:
        f.write(folder + "\n")
