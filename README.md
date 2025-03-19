# Video Structuring

环境配置：`pip install -r requirements.txt`

视频关键帧提取代码：
```
python py_scene_detect.py
```

视频结构化代码：
```
python track_on_videos_multi.py
```

`SCENE_GRAPH_PATH`指定的是视频三元组路径，由gpt-3.5生成；`VIDEO_DIR`指定的是视频的帧数据文件夹，是由`py_scene_detect.py`运行得到的目录；视频中目标的clip特征会保存到指定的`visual_embeddings`路径，输出的json文件保存到`output_filepath`指定的路径。