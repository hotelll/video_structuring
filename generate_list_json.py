import os
import json

target_dir = "/home/hetianyao/Video-LLaVA-PG/datasets/ActivityNet_QA/test"
file_paths = [os.path.join(target_dir, file) for file in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, file))]
output_file = "/home/hetianyao/Video-LLaVA-PG/datasets/ActivityNet_QA/video_list2.json"
with open(output_file, "w") as json_file:
    json.dump(file_paths, json_file, indent=4)

print(f"已将文件路径保存到 {output_file}")
