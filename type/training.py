from typing import TypedDict

RenderingOptions = TypedDict('RenderingOptions', 
                             {'warping_mask': str, #none, cube, mesh
                              'cfg_name': str, #aist surreal_new surreal aist_rescaled shhq deepfashion
                              'year': int,
                              'canon_logging': bool,
                              'cam2world': any, #'camera_params[0]', 
                              'intrinsics': any, # camera_params[1] 
                              'depth_resolution': any,
                              'ray_start': any, 
                              'ray_end': any,
                              'disparity_space_sampling': any,
                              'projector': str, #none
                              'box_warp_pre_deform': any,
                              'depth_resolution_importance': any,
                              'triplane_depth': int | None ,
                              'density_noise': int,
                              'project_inside_only': bool,
                              'mesh_clip_offset':int, # 0.05 
                              'decoder_lr_mul': int
                              }
                              )
