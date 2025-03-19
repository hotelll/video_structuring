import os
import cv2
from tqdm import tqdm
video_list_file = "/home/hetianyao/Video-LLaVA-PG/empty_folders_act_test.txt"
video_directory = "/home/hetianyao/Video-LLaVA-PG/datasets/ActivityNet_QA/test"
output_directory = "/mnt/sdb/dataset/ActivityNet_QA_test/video_frames3"

os.makedirs(output_directory, exist_ok=True)

with open(video_list_file, "r") as file:
    video_names = [line.strip() for line in file if line.strip()]

for video_name in tqdm(video_names):
    video_path = os.path.join(video_directory, f"{video_name}.mp4")
    if not os.path.exists(video_path):
        print(f"视频文件未找到：{video_path}")
        continue
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < 8:
        print(f"视频帧数不足8帧，跳过：{video_path}")
        cap.release()
        continue

    frame_indices = [int(frame_count * i / 8) for i in range(8)]
    output_folder = os.path.join(output_directory, video_name)
    os.makedirs(output_folder, exist_ok=True)

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_folder, f"scene_{idx}.png")
            cv2.imwrite(output_path, frame)
        else:
            print(f"读取帧失败：{video_path}, 帧索引：{frame_idx}")

    cap.release()
    print(f"完成视频处理：{video_name}")

for video_name in video_names:
    video_path = os.path.join(video_directory, f"{video_name}.mkv")
    if not os.path.exists(video_path):
        print(f"视频文件未找到：{video_path}")
        continue
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < 8:
        print(f"视频帧数不足8帧，跳过：{video_path}")
        cap.release()
        continue

    frame_indices = [int(frame_count * i / 8) for i in range(8)]
    output_folder = os.path.join(output_directory, video_name)
    os.makedirs(output_folder, exist_ok=True)

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_folder, f"scene_{idx}.png")
            cv2.imwrite(output_path, frame)
        else:
            print(f"读取帧失败：{video_path}, 帧索引：{frame_idx}")

    cap.release()
    print(f"完成视频处理：{video_name}")

print("所有视频处理完成！")
