import json
from tqdm import tqdm
import copy
import os


SCENE_GRAPH_PATH = "datasets/videochatgpt/train_json/scene_graph_filt.json"
OUTPUT_PATH = "datasets/videochatgpt/train_json/scene_graph_filt_newform.json"


if __name__ == "__main__":
    sg_data = json.load(open(SCENE_GRAPH_PATH))

    new_sg_data = []
    for video_name, triplets in tqdm(sg_data.items()):
        new_sample = {
            "video_name": video_name.split(".")[0],
            "triplets": triplets
        }
        new_sg_data.append(new_sample)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(new_sg_data, f, indent=4)
