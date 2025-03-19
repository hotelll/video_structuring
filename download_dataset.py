from datasets import load_dataset


# /home/hetianyao/Video-LLaVA-PG/README.md

subdir_list = [ 
                # '0_30_s_academic_v0_1', 
                # '0_30_s_youtube_v0_1', 
                # '0_30_s_activitynet',  # not downloaded
                # '0_30_s_perceptiontest', 
                # '0_30_s_nextqa', 
                # '30_60_s_academic_v0_1', 
                # '30_60_s_youtube_v0_1', 
                # '30_60_s_activitynet',  # not downloaded
                # '30_60_s_perceptiontest', 
                # '30_60_s_nextqa', 
                # '1_2_m_youtube_v0_1', 
                # '1_2_m_academic_v0_1', 
                # '1_2_m_activitynet', # not downloaded
                # '1_2_m_nextqa', 
                # '2_3_m_youtube_v0_1',
                # '2_3_m_academic_v0_1', 
                # '2_3_m_activitynet', # not downloaded
                '2_3_m_nextqa', 
                'llava_hound'
            ]


for subdir in subdir_list:
    llavaVideoDataset = load_dataset("lmms-lab/LLaVA-Video-178K", subdir)
    llavaVideoDataset.save_to_disk("/mnt/sde/LLaVA-Video-178K/{}".format(subdir))