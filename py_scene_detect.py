from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
import cv2
import os
from tqdm import tqdm
import json

def extract_middle_frame(video, video_path, scene_list, output_dir):
    # 初始化视频管理器
    # 遍历所有场景，提取每个场景的中间帧
    if scene_list == []:
        # i = 0
        video_capture = cv2.VideoCapture(video_path)
        middle_frame_number = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
        video.seek(middle_frame_number)  # 跳转到该帧
        frame = video.read()  # 读取该帧
        output_path = os.path.join(output_dir, f"scene_{i + 1}.png")
        cv2.imwrite(output_path, frame)  # 保存关键帧
    else:
        for i, (start_frame, end_frame) in enumerate(scene_list):
            middle_frame_number = (start_frame.get_frames() + end_frame.get_frames()) // 2  # 计算中间帧号
            video.seek(middle_frame_number)  # 跳转到该帧
            frame = video.read()  # 读取该帧

            output_path = os.path.join(output_dir, f"scene_{i + 1}.png")
            cv2.imwrite(output_path, frame)  # 保存关键帧
            # print(f"Keyframe for Scene {i + 1} saved to {output_path}.")


def detect_scenes_with_adaptive_threshold(video_path, output_dir, min_scenes=1, max_scenes=10):
    # low, high = 10.0, 100.0  # 阈值范围
    # best_threshold = None
    # scene_list = []
    #
    # search_iter = 0
    # # 二分搜索以找到适合的阈值
    # while low <= high:
    #     if search_iter > 10:
    #         break
    #     mid = (low + high) // 2
    #     # print(f"Trying threshold: {mid}")
    #
    #     # 使用中间值作为阈值创建检测器
    #     # scene_manager.clear()  # 清除之前的检测结果
    #     scene_manager = SceneManager()
    #     video = open_video(video_path)
    #     scene_manager.add_detector(ContentDetector(threshold=mid))
    #
    #     # 检测场景
    #     scene_manager.detect_scenes(video)
    #     scene_list = scene_manager.get_scene_list()
    #
    #     # print(f"Detected {len(scene_list)} scenes with threshold {mid}.")
    #
    #     # 根据检测到的场景数量调整搜索范围
    #
    #     if len(scene_list) < min_scenes:
    #         high = mid - 0.1  # 阈值太高，减少它
    #     elif len(scene_list) > max_scenes:
    #         low = mid + 0.1  # 阈值太低，增加它
    #     else:
    #         best_threshold = mid  # 找到合适的阈值
    #         break
    #     search_iter += 1

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 提取每个场景的中间帧
    # extract_middle_frame(video, video_path, scene_list, output_dir)


# 示例调用：检测视频场景并提取关键帧
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    main_out_dir = "/mnt/sdb/dataset/ActivityNet_QA_test/video_frames"
    with open("/home/hetianyao/Video-LLaVA-PG/datasets/ActivityNet_QA/video_list2.json", "r") as f:
        video_list = json.load(f)
    finished_list = os.listdir("/mnt/sdb/dataset/ActivityNet_QA_test/video_frames")
    all_list = [os.path.basename(video_path).split('.')[0] for video_path in video_list]
    unfinished_list = list(set(all_list) - set(finished_list))
    video_list = [video_path for video_path in video_list if os.path.basename(video_path).split('.')[0] in unfinished_list]
    for video_path in tqdm(video_list[1:]):
        try:
            video_name = os.path.basename(video_path).split('.')[0]
            output_dir = os.path.join(main_out_dir, video_name)
            detect_scenes_with_adaptive_threshold(video_path, output_dir, min_scenes=3, max_scenes=10)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
    # detect_scenes_and_extract_frames(video_path, output_dir, threshold=50)
