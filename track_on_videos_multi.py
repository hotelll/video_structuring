import os

rank = int(os.environ.get('RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '1'))
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('RANK', '0')

import sys

sys.path.append('/home/hetianyao/Video-LLaVA-PG')
import argparse

from video_chatgpt.utils import disable_torch_init
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import *
from grounding_evaluation.grounding_new_api import Tracker_with_GroundingDINO
from grounding_evaluation.grounding_new_api import cfg as default_cfg

import numpy as np
import json
from PIL import Image, ImageFile
from tqdm import tqdm

import os

# os.environ["OPENAI_API_KEY"] = "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"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch

disable_torch_init()

SCENE_GRAPH_PATH = "datasets/videochatgpt/train_json/scene_graph_filt_newform.json"
VIDEO_DIR = "datasets/videochatgpt/video_frames"

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--projection_path", type=str, required=True)
    parser.add_argument("--use_asr", action='store_true', help='Whether to use audio transcripts or not')
    parser.add_argument("--conv_mode", type=str, required=False, default='pg-video-llava')
    parser.add_argument("--with_grounding", action='store_true', help='Run with grounding module')

    args = parser.parse_args([
        "--model-name", "weights/llava/llava-v1.5-7b",
        "--projection_path", "weights/projection/mm_projector_7b_1.5_336px.bin",
        # "--use_asr",
        "--conv_mode", "pg-video-llava",
        "--with_grounding"
    ])

    return args


if __name__ == '__main__':
    
    with open(SCENE_GRAPH_PATH, "r") as f:
        video_sgs = json.load(f)
    # torch.set_default_device(rank)
    class_id_dict = {}
    for video in video_sgs:
        video_key = video["video_name"]
        triplet_list = video["triplets"]
        class_list = []
        class_list = set(class_list)
        for triplet in triplet_list:
            if len(triplet) == 3:
                if triplet[0] not in ["", "video"]:
                    class_list.add(triplet[0])
                if triplet[2] not in ["", "video"]:
                    class_list.add(triplet[2])
            elif len(triplet) == 2:
                if triplet[0] not in ["", "video"]:
                    class_list.add(triplet[0])
            else:
                continue
        class_id_dict[video_key] = list(class_list)

    tracker = Tracker_with_GroundingDINO(
        config=default_cfg, deva_model_path=default_cfg['deva_model_path'],
        temporal_setting='online',
        detection_every=1,
        max_missed_detection_count=1,
        max_num_objects=-1,  # TODO change
    )

    video_subdirs = [os.path.join(VIDEO_DIR, subdir) for subdir in os.listdir(VIDEO_DIR)]
    # video_subdirs = video_subdirs[:100]
    video_subdirs.sort()
    output_dict = {}

    for subdir in tqdm(video_subdirs[rank::world_size]):
        try:
            video_name = subdir.split('/')[-1]
            frame_list = []

            id_list = sorted([int(filename.split('_')[1].split('.')[0]) for filename in os.listdir(subdir)])
            for scene_id in id_list:
                frame_path = os.path.join(subdir, 'scene_{}.png'.format(scene_id))
                frame_list.append(frame_path)

            img_list = []
            for frame_path in frame_list:
                frame = np.array(Image.open(frame_path).convert("RGB"))
                img_list.append(frame)
            video_key = os.path.basename(subdir)
            class_list = class_id_dict[video_key]
            tracking_results = []
            frame_num = len(img_list)

            for class_label in class_list:
                tracking_result = tracker.run_on_list_of_images(img_list, [class_label])
                tracking_results.append(tracking_result)
            frame_res_list = []
            for i in range(frame_num):  # iterate over scenes
                frame_list = []
                obj_start_id = 0

                img_clip_list = []
                comb_tmp_id_to_obj = {}
                comb_all_obj_ids = []
                comb_prompts = []

                for class_id, class_label in enumerate(class_list):  # iterate over classes
                    class_j_in_frame_i = tracking_results[class_id][i]
                    tmp_id_to_obj = class_j_in_frame_i['tmp_id_to_obj']
                    all_obj_ids = class_j_in_frame_i['all_obj_ids']
                    prompts = class_j_in_frame_i['prompts']
                    image_clip_features = None
                    if len(tmp_id_to_obj) > 0:
                        image_clip_features = torch.cat([v.clip_features[0] for k, v in tmp_id_to_obj.items()], dim=0)

                    # image_clip_features = class_j_in_frame_i['image_clip_features']
                    if image_clip_features != None:
                        img_clip_list.append(image_clip_features)

                        if len(image_clip_features) != len(tmp_id_to_obj):
                            print(1)

                        obj_ids = [i for i in tmp_id_to_obj]
                        for cur_obj_id in obj_ids:
                            comb_obj_id = obj_start_id + cur_obj_id
                            tmp_id_to_obj_sample = {
                                'category_id': class_id,
                                'id': tmp_id_to_obj[cur_obj_id].id,
                                'score': str(tmp_id_to_obj[cur_obj_id].scores[0]),
                                # 'clip_feature': tmp_id_to_obj[cur_obj_id].clip_feature
                            }
                            comb_tmp_id_to_obj[comb_obj_id] = tmp_id_to_obj_sample

                    comb_all_obj_ids += all_obj_ids
                    comb_prompts += prompts
                    obj_start_id += len(all_obj_ids)

                embed_dir = '/'.join(subdir.split('/')[:-2] + ['visual_embeddings'])
                embed_dir = os.path.join(embed_dir, video_name)
                os.makedirs(embed_dir, exist_ok=True)
                embed_path = os.path.join(embed_dir, 'scene_{}.pt'.format(i + 1))
                if len(img_clip_list) > 0:
                    comb_embeds = torch.cat(img_clip_list, dim=0)
                    torch.save(comb_embeds, embed_path)

                frame_info = {
                    'embed_dir': embed_dir,
                    'tmp_id_to_obj': comb_tmp_id_to_obj,

                    'all_obj_ids': comb_all_obj_ids,
                    'prompts': comb_prompts
                }
                frame_res_list.append(frame_info)

            output_dict[video_name] = frame_res_list
        except:
            print('Error in video: {}'.format(video_name))
    
    os.makedirs("/home/hetianyao/Video-LLaVA-PG/temporal", exist_ok=True)
    with open('/home/hetianyao/Video-LLaVA-PG/temporal/test_video_structure_v1_{}.json'.format(rank), 'w') as f:
        json.dump(output_dict, f, indent=4)


def merge_json_files(input_dir, output_file):
    merged_data = {}
    json_files = [f for f in os.listdir(input_dir) if f.startswith('test_video_structure_v1_') and f.endswith('.json')]
    json_files.sort()
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            merged_data.update(data)
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

    print(f"Merged {len(json_files)} files into {output_file}")


input_directory = "/home/hetianyao/Video-LLaVA-PG/temporal"
output_filepath = os.path.join(input_directory, 'test_video_structure_v1.json')
merge_json_files(input_directory, output_filepath)