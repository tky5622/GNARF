import torch

def generate_smpl_translate(smpl_params, smpl_avg_body_pose, smpl_avg_transl):
    smpl_orient = smpl_params[:, :3]
    # smpl_body_pose = smpl_params[:, 3:72]
    # smpl_betas = smpl_params[:, 72:82]
    if smpl_params.shape[-1] > 82:
        smpl_translate = smpl_params[:, 82:85]
        # smpl_translate = torch.zeros_like(smpl_params[:, 82:85])
    else:
        batch_size = smpl_orient.shape[0]
        bs_expand = batch_size if smpl_avg_body_pose.shape[0] == 1 else -1
        smpl_translate = smpl_avg_transl.expand(bs_expand, -1)
    return smpl_translate

def generate_smpl_sample(
        rendering_options,
        ray_directions,
        get_ray_limits_box,
        ray_origins,
        smpl_orient,
        smpl_avg_body_pose,
        smpl_clip,
        smpl_avg_betas,
        smpl_betas,
        smpl_body_pose,
        smpl_avg_scale,
        camera_params,
        get_smpl_min_max_depth,
):
    if rendering_options['warping_mask'] == 'none':
        sample_mask = None
    elif rendering_options['warping_mask'] == 'cube':
        if torch.isnan(ray_directions[0, 0, 0]).item():
            # ray_start, ray_end = rendering_options['ray_start'], rendering_options['ray_end']
            ray_start, ray_end = 0., 1.
            sample_mask = None
        else:
            # -1,1 cube
            ray_start, ray_end = get_ray_limits_box(ray_origins, ray_directions, rendering_options)
            # Handle invalid cases.
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            sample_mask = is_ray_valid[...,None,:].expand(-1, -1, rendering_options['depth_resolution'], -1).reshape(is_ray_valid.shape[0], -1)
    elif rendering_options['warping_mask'] == 'mesh':
        if torch.isnan(ray_directions[0, 0, 0]).item():
            ray_start, ray_end = rendering_options['ray_start'], rendering_options['ray_end']
            sample_mask = None
        else:
            batch_size = smpl_orient.shape[0]
            bs_expand = batch_size if smpl_avg_body_pose.shape[0] == 1 else -1
            if rendering_options.get('canon_logging', False):
                smpl_clip_current = smpl_clip(betas=smpl_avg_betas.expand(bs_expand, -1), body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1), global_orient=self.smpl_avg_orient.expand(bs_expand, -1), transl=self.smpl_avg_transl.expand(bs_expand, -1))
            else:
                smpl_clip_current = smpl_clip(betas=smpl_betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=smpl_translate)
            smpl_clip_current.vertices *= smpl_avg_scale
            rendering_options.update({'cam2world': camera_params[0], 'intrinsics': camera_params[1]}) # bs x 4 x 4, bs x 3 x 3
            ray_start, ray_end = get_smpl_min_max_depth(rendering_options, ray_directions, smpl_clip_current.vertices, self.smpl_clip.faces_t)
            sample_mask = (ray_start[..., None, :] <= ray_end[..., None, :]).expand(-1, -1, rendering_options['depth_resolution'], -1).reshape(batch_size, -1)
            if not sample_mask.max(dim=-1)[0].all():
                sample_mask = None
                    # print(f'Warning, out of frame SMPL mesh detected!')
    else:
        raise NotImplementedError()
    return sample_mask, ray_start, ray_end

def generate_depths_coarse(
        rendering_options, 
        ray_origins,
        sample_stratified,
        ray_start, ray_end
        ):
    if rendering_options['warping_mask'] == 'cube' or rendering_options['warping_mask'] == 'mesh':
        depths_coarse = sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
    else:
        depths_coarse = sample_stratified(
            ray_origins, 
            rendering_options['ray_start'], 
            rendering_options['ray_end'], 
            rendering_options['depth_resolution'], 
            rendering_options['disparity_space_sampling'])
    return depths_coarse

def panic3d_rendering(
        rendering_options,
        math_utils,
        ray_origins,
        ray_directions,
        sample_stratified #self
):
    if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
        ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
        is_ray_valid = ray_end > ray_start
    if torch.any(is_ray_valid).item():
        ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
        ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
        depths_coarse = sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
    else:
    # Create stratified depth samples
        depths_coarse = sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], 
                                               rendering_options['disparity_space_sampling'])
    return depths_coarse



def change_for_eric_model(ray_directions, 
                          rendering_options, 
                          smpl_reduced, 
                          smpl_betas, 
                          smpl_body_pose,
                          smpl_avg_body_pose,
                          smpl_avg_betas,
                          smpl_avg_orient,
                          bs_expand,
                          smpl_avg_transl,
                          smpl_translate,
                          smpl_avg_scale,
                          get_canonical_coordinates,
                          sample_mask,
                          warp_grid,
                          needs_clip_mask,
                          smpl_clip,
                          camera_params,
                          get_smpl_min_max_depth
                          ):
    # change to be compatible with Eric's new models
    if torch.isnan(ray_directions[0, 0, 0]).item() and rendering_options['projector'] == 'none':
        smpl_reduced_current = None
        smpl_reduced_canon = None
    else:
        smpl_reduced_current = smpl_reduced(betas=smpl_betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=smpl_translate)
        smpl_reduced_canon = smpl_reduced(betas=smpl_avg_betas.expand(bs_expand, -1),
                                                   body_pose=smpl_avg_body_pose.expand(bs_expand, -1),
                                                   global_orient=smpl_avg_orient.expand(bs_expand, -1),
                                                   transl=smpl_avg_transl.expand(bs_expand, -1))
        smpl_reduced_current.transl = smpl_translate
        smpl_reduced_canon.transl = smpl_avg_transl.expand(bs_expand, -1)
        smpl_reduced_current.vertices *= smpl_avg_scale
        smpl_reduced_canon.vertices *= smpl_avg_scale

    if rendering_options['box_warp_pre_deform']:
        sample_coordinates = (2 / rendering_options['box_warp']) * sample_coordinates

    sample_coordinates = get_canonical_coordinates(
            sample_coordinates,
            mask=sample_mask,
            warp_field=warp_grid,
            smpl_src=smpl_reduced_current,
            smpl_dst=smpl_reduced_canon,
            projector=rendering_options['projector']
        )

    # Precompute clip depths
    smpl_clip_depths = None
    if needs_clip_mask(rendering_options):
        if rendering_options.get('canon_logging', False):
            smpl_clip_current = smpl_clip(betas=smpl_avg_betas.expand(bs_expand, -1), body_pose=self.smpl_avg_body_pose.expand(bs_expand, -1),
                                                   global_orient=smpl_avg_orient.expand(bs_expand, -1), transl=self.smpl_avg_transl.expand(bs_expand, -1))
        else:
            smpl_clip_current = smpl_clip(betas=smpl_betas, body_pose=smpl_body_pose, global_orient=smpl_orient, transl=smpl_translate)
        smpl_clip_current.vertices *= smpl_avg_scale
        if rendering_options['warping_mask'] == 'mesh':
            rendering_options.update({'cam2world': camera_params[0], 'intrinsics': camera_params[1]}) # bs x 4 x 4, bs x 3 x 3
        smpl_clip_depths = get_smpl_min_max_depth(rendering_options, ray_directions, smpl_clip_current.vertices, self.smpl_clip.faces_t)
    return sample_coordinates, sample_mask, smpl_reduced_current, smpl_reduced_canon, smpl_clip_depths


def mask_out_invalid_samples(smpl_clip_depths, get_sample_mask, depths_coarse):
            # Mask out invalid samples (optional).
        is_sample_valid = None
        if smpl_clip_depths is not None:
            is_sample_valid = get_sample_mask(sample_depths=depths_coarse, min_max_depths=smpl_clip_depths)
            densities_coarse = densities_coarse - 1000 * (1-is_sample_valid.float())
        return is_sample_valid, densities_coarse

def mask_out_clip_depth(smpl_clip_depths):
    # Mask out invalid samples (optional).
    if smpl_clip_depths is not None:
        is_sample_valid = self.get_sample_mask(sample_depths=depths_fine, min_max_depths=smpl_clip_depths)
        densities_fine = densities_fine - 1000 * (1-is_sample_valid.float())
        #colors_fine = colors_fine * is_sample_valid.float()


def generate_finals(
        rendering_options,
        ray_marcher, #self
        colors_coarse,
        densities_coarse,
        depths_coarse,
        sample_importance, #self
        ray_origins,
        ray_directions,
        get_canonical_coordinates, #self
        sample_mask,
        warp_grid,
        smpl_reduced_current,
        smpl_reduced_canon,
        run_model, #self
        planes,
        decoder,
        triplane_crop,
        triplane_crop_mask,
        binarize_clouds,
        cull_clouds_mask,
        cull_clouds,
        batch_size,
        num_rays,
        unify_samples, #self
        xyz_coarse,
        ):
    N_importance = rendering_options['depth_resolution_importance']
    if N_importance > 0:
        _, _, weights = ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

        depths_fine = sample_importance(depths_coarse, weights, N_importance)
        #GNARF
        # sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            
        # shu added: have all directions point to center (for ambient-lit)
        if rendering_options.get('directionless', False):
            sample_directions = -sample_coordinates/sample_coordinates.norm(dim=-1, keepdim=True).clip(0.01)
        # this option is for GNARF
        elif rendering_options['box_warp_pre_deform']:
            sample_coordinates = (2 / rendering_options['box_warp']) * sample_coordinates
            sample_coordinates = get_canonical_coordinates(
            sample_coordinates,
            mask=sample_mask,
            warp_field=warp_grid,
            smpl_src=smpl_reduced_current,
            smpl_dst=smpl_reduced_canon,
            projector=rendering_options['projector']
        )
        else:
            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)

        out = run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)
        colors_fine = out['rgb']
        densities_fine = out['sigma']
        ##panic3d
        xyz_fine = out['xyz']
        if triplane_crop:
                # print(xyz_fine.amin(dim=(0,1)))
                # print(xyz_fine.amax(dim=(0,1)))
            cropmask = triplane_crop_mask(xyz_fine, triplane_crop, rendering_options['box_warp'])
            densities_fine[cropmask] = -1e3
        if binarize_clouds:
            ccmask = cull_clouds_mask(densities_fine, binarize_clouds)
            densities_fine[ccmask] = -1e3
            densities_fine[~ccmask] = 1e3
        elif cull_clouds:
            ccmask = cull_clouds_mask(densities_fine, cull_clouds)
            densities_fine[ccmask] = -1e3
        #end
        colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
        densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
        #panic3d
        xyz_fine = xyz_fine.reshape(batch_size, num_rays, N_importance, xyz_fine.shape[-1])
        #mask_out_clip_depth()
        #original GNARF code
        #all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse,
        #all_xyz, depths_coarse, colors_coarse, densities_coarse, xyz_coarse xyz_fine
        all_depths, all_colors, all_densities, all_xyz = unify_samples(
            depths_coarse, colors_coarse, densities_coarse, xyz_coarse,
            depths_fine, colors_fine, densities_fine, xyz_fine,
        )

            # Aggregate
            #GNARF
            # rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        all_colors_ = torch.cat([all_colors, all_xyz], dim=-1)
        rgb_final_, depth_final, weights = ray_marcher(all_colors_, all_densities, all_depths, rendering_options)
        rgb_final = rgb_final_[...,:-3]
        xyz_final = rgb_final_[...,-3:]
    else:
            #GNARF
            # rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
        colors_coarse_ = torch.cat([colors_coarse, xyz_coarse], dim=-1)
        rgb_final_, depth_final, weights = ray_marcher(colors_coarse_, densities_coarse, depths_coarse, rendering_options)
        rgb_final = rgb_final_[...,:-3]
        xyz_final = rgb_final_[...,-3:]
    return rgb_final, xyz_final, depth_final, weights



#panic3d
def calc_mask(
        triplane_crop, 
        triplane_crop_mask, 
        xyz_coarse,
        densities_coarse,
        binarize_clouds,
        cull_clouds_mask,
        cull_clouds,
        rendering_options
        ):
    if triplane_crop:
        # print(xyz_fine.amin(dim=(0,1)))
        # print(xyz_fine.amax(dim=(0,1)))
        cropmask = triplane_crop_mask(xyz_coarse, triplane_crop, rendering_options['box_warp'])
        densities_coarse[cropmask] = -1e3
    if binarize_clouds:
        ccmask = cull_clouds_mask(densities_coarse, binarize_clouds)
        densities_coarse[ccmask] = -1e3
        densities_coarse[~ccmask] = 1e3
    elif cull_clouds:
        ccmask = cull_clouds_mask(densities_coarse, cull_clouds)
        densities_coarse[ccmask] = -1e3
    return ccmask, densities_coarse, cropmask

        # if triplane_crop:
        #     cropmask = triplane_crop_mask(xyz_coarse, triplane_crop, rendering_options['box_warp'])
        #     densities_coarse[cropmask] = -1e3
        # print(out['rgb'].shape)
        # print(out['sigma'].shape)
