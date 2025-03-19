import sys
sys.path.append('/home/hetianyao/Video-LLaVA-PG')
import argparse
from video_chatgpt.video_conversation import (default_conversation)
from video_chatgpt.video_conversation import load_video
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
from video_chatgpt.audio_transcript.transcribe import Transcriber

from video_chatgpt.utils import disable_torch_init
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import *
from grounding_evaluation.grounding_new_api import Tracker_with_GroundingDINO
from grounding_evaluation.grounding_new_api import cfg as default_cfg 
from grounding_evaluation.util.image_tagging import TaggingModule, get_unique_tags
from grounding_evaluation.util.entity_matching_openai import EntityMatchingModule
import subprocess
import glob
import tempfile
import random
import string
import datetime
import os
os.environ["OPENAI_API_KEY"] = "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"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
disable_torch_init()


class PGVideoLLaVA():
    def __init__(self, args_model_name, args_projection_path, use_asr=False, conv_mode="pg-video-llava", temperature=0.2, max_output_tokens=1024) -> None:
        super().__init__(args_model_name, args_projection_path, use_asr, conv_mode, temperature, max_output_tokens)
        self.tracker = Tracker_with_GroundingDINO(
            config=default_cfg, deva_model_path=default_cfg['deva_model_path'], 
                    temporal_setting='online',
                    # temporal_setting=None,
                    detection_every=5,
                    max_missed_detection_count=1,
                    max_num_objects=5, #TODO change
                    # dino_threshold=0.35
            )
        self.tagging_model = TaggingModule()
        self.entity_match_module = EntityMatchingModule()
        
    def answer(self, with_grounding=True, output_dir='outputs'):
        # Run the video-based LMM
        llm_output = super().answer()
        if not with_grounding:
            return llm_output
        
        # Apply image-tagging model
        tags_in_video = self.tagging_model.run_on_video(self.video_frames_pil)
        classes = get_unique_tags(tags_in_video)
        entity_list = classes[:10]
        # Apply entity matching model
        highlight_output, match_state = self.entity_match_module(llm_output, entity_list)
        class_list = list(set(match_state.values()))
        
        # Split the video
        print('Splitting into segments ...')
        temp_dir_splits = tempfile.TemporaryDirectory()
        temp_dir_saves = tempfile.TemporaryDirectory()
        temp_dir_splits.name = "tmp/split_scenes/"
        temp_dir_saves.name = "tmp/split_saves/"
        _ = subprocess.call(["scenedetect", "-i", self.video_path, "split-video", "-o", temp_dir_splits.name])

        # For each split run tracker
        print('Running tracker in each segment ...')
        for video_split_name in os.listdir(temp_dir_splits.name):
            # self.tracker.run_on_video(os.path.join(temp_dir_splits.name, video_split_name), os.path.join(temp_dir_saves.name,video_split_name.rsplit('.', 1)[0] + ".avi"), class_list)
            self.tracker.run_on_video(os.path.join(temp_dir_splits.name, video_split_name), os.path.join(temp_dir_saves.name,video_split_name), class_list)
            
        print('Combining output videos ...')
        # Output file path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        _random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        output_video_path = os.path.join(output_dir, f"video_{_timestamp}_{_random_chars}.mp4")
        output_video_path_h264 = os.path.join(output_dir, f"video_{_timestamp}_{_random_chars}_h264.mp4")
        # Combine splits and save
        mp4_files = sorted(glob.glob(os.path.join(temp_dir_saves.name, '*.mp4')))
        mp4_files = [os.path.basename(mp4_file) for mp4_file in mp4_files]
        txt = ''
        with open(os.path.join(temp_dir_saves.name, 'video_list.txt'), 'w') as file:
            for mp4_file in mp4_files:
                txt+= "file "+ mp4_file + '\n'
            file.write(txt)
        _ = subprocess.run([f"ffmpeg -f concat -safe 0 -i {temp_dir_saves.name}/video_list.txt -c copy {output_video_path} -y"], shell=True)
        _ = subprocess.run([f"ffmpeg -i {output_video_path} -vcodec libx264 {output_video_path_h264} -y"], shell=True)
        # os.system("ffmpeg -i Video.mp4 -vcodec libx264 Video2.mp4")
        _ = subprocess.run([f"rm {temp_dir_saves.name}/video_list.txt"], shell=True)
        
        temp_dir_splits.cleanup()
        temp_dir_saves.cleanup()
        
        return llm_output, output_video_path_h264, highlight_output, match_state

    def interact(self):
        print("Welcome to PG-Video-LLaVA !")
        
        video_set=False
        
        while True:
            if not video_set:
                video_path = input("Please enter the video file path: ")
                self.upload_video(video_path)
                video_set = True
                
            try:
                text = input("USER>>")
                if not text:
                    print('----------\n\n')
                    self.clear_history()
                    video_set = False
                    continue
                    # return
                
                self.add_text(text, video_path)
                llm_output, output_video_path_h264, highlight_output, match_state = self.answer(with_grounding=True) # type: ignore
                print('ASSISTANT>>', llm_output)
                print('\nGROUNDING>>', '\t', output_video_path_h264, '\n\t', match_state, '\n')
            
            except KeyboardInterrupt:
                # self.clear_history()
                print('----------')
                print('QUITTING...')
                return

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
        "--conv_mode","pg-video-llava",
        "--with_grounding"
    ])

    return args

if __name__=='__main__':
    args = parse_args()
    
    if args.with_grounding:
        chat = PGVideoLLaVA(
            args_model_name=args.model_name,
            args_projection_path=args.projection_path,
            use_asr=args.use_asr, 
            conv_mode=args.conv_mode,
        )
        chat.interact()
    else:
        chat = VideoChatGPTInterface(
            args_model_name=args.model_name,
            args_projection_path=args.projection_path,
            use_asr=args.use_asr, 
            conv_mode=args.conv_mode,
        )
        chat.interact()
    
    