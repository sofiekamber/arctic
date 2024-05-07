import torch
import torch.nn as nn

import common.torch_utils as torch_utils
from common.torch_utils import nanmean
import coap
from common.body_models import build_mano_coap

l1_loss = nn.L1Loss(reduction="none")
mse_loss = nn.MSELoss(reduction="none")


def subtract_root_batch(joints: torch.Tensor, root_idx: int):
    assert len(joints.shape) == 3
    assert joints.shape[2] == 3
    joints_ra = joints.clone()
    root = joints_ra[:, root_idx : root_idx + 1].clone()
    joints_ra = joints_ra - root
    return joints_ra


def compute_contact_devi_loss(pred, targets):
    cd_ro = contact_deviation(
        pred["object.v.cam"],
        pred["mano.v3d.cam.r"],
        targets["dist.ro"],
        targets["idx.ro"],
        targets["is_valid"],
        targets["right_valid"],
    )

    cd_lo = contact_deviation(
        pred["object.v.cam"],
        pred["mano.v3d.cam.l"],
        targets["dist.lo"],
        targets["idx.lo"],
        targets["is_valid"],
        targets["left_valid"],
    )
    cd_ro = nanmean(cd_ro)
    cd_lo = nanmean(cd_lo)
    cd_ro = torch.nan_to_num(cd_ro)
    cd_lo = torch.nan_to_num(cd_lo)
    return cd_ro, cd_lo


def contact_deviation(pred_v3d_o, pred_v3d_r, dist_ro, idx_ro, is_valid, _right_valid):
    right_valid = _right_valid.clone() * is_valid
    contact_dist = 3 * 1e-3  # 3mm considered in contact
    vo_r_corres = torch.gather(pred_v3d_o, 1, idx_ro[:, :, None].repeat(1, 1, 3))

    # displacement vector H->O
    disp_ro = vo_r_corres - pred_v3d_r  # batch, num_v, 3
    invalid_ridx = (1 - right_valid).nonzero()[:, 0]
    disp_ro[invalid_ridx] = float("nan")
    disp_ro[dist_ro > contact_dist] = float("nan")
    cd = (disp_ro**2).sum(dim=2).sqrt()
    err_ro = torch_utils.nanmean(cd, axis=1)  # .cpu().numpy()  # m
    return err_ro

def compute_coap_loss(pred):
    #print(pred["mano.pose.r"].shape)
    #print(pred.keys())
    #assert (False)

    coap_loss_r = coap_loss(pred["object.v.cam"], pred["mano.beta.r"], pred["mano.pose.r"], is_right=True)
    coap_loss_l = coap_loss(pred["object.v.cam"], pred["mano.beta.l"], pred["mano.pose.l"], is_right=False)



    return coap_loss_r, coap_loss_l


def coap_loss(pred_v3d_object, pred_betas_r, pred_rotmat_r, is_right):
    model = build_mano_coap(is_right)

    device = "cuda:0" if torch.cuda.is_available() else "cpu" #TODO: clean this
    model = coap.attach_coap(model, pretrained=True, device=device) #Need model with pca


    #prep data
    torch_param = {}
    smpl_body_pose = torch.zeros((1, 48), dtype=torch.float, device=device)
    #smpl_body_pose[:, :48] = pred_rotmat_r
    torch_param['global_orient'] = torch.zeros(8, 3, device='cuda:0', requires_grad=True) #Needed for correct dimensionality!

    torch_param['hand_pose'] = torch.zeros((8, 48), dtype=torch.float, device=device) #smpl_body_pose.to(torch.float32) #TODO: convert rotation matrix to vector
    torch_param['betas'] = pred_betas_r
    #torch_param['transl'] = torch.from_numpy(np.array([[-5.0101800e-003, 1.5031957e-001, 3.0754370e-002]])).to(torch.float32).to(args.device)


    mano_output = model(**torch_param, return_verts=True, return_full_pose=True)
    assert model.joint_mapper is None, 'COAP requires valid SMPL joints as input'

    scene_points = sample_scene_points(model, mano_output, pred_v3d_object) #TODO: implement sample_scene_points for batch size > 1
    #scene_points = pred_v3d_object
    #scene_points = torch.zeros((8, 50, 3), dtype=torch.float, device=device)
    coap_loss, _collision_mask = model.coap.collision_loss(scene_points, mano_output, ret_collision_mask=True)

    return coap_loss


@torch.no_grad()
def sample_scene_points(model, smpl_output, scene_vertices, scene_normals=None, n_upsample=2, max_queries=10000):
    points = scene_vertices.clone()
    # remove points that are well outside the SMPL bounding box
    bb_min = smpl_output.vertices.min(1).values.reshape(1, 3)
    bb_max = smpl_output.vertices.max(1).values.reshape(1, 3)

    inds = (scene_vertices >= bb_min).all(-1) & (scene_vertices <= bb_max).all(-1)
    if not inds.any():
        return None
    points = scene_vertices[inds]
    model.coap.eval()
    colliding_inds = (model.coap.query(points.reshape((1, -1, 3)), smpl_output) > 0.5).reshape(-1)
    model.coap.detach_cache()  # detach variables to enable differentiable pass in the opt. loop
    if not colliding_inds.any():
        return None
    points = points[colliding_inds.reshape(-1)]

    if scene_normals is not None and points.size(0) > 0:  # sample extra points if normals are available
        norms = scene_normals[inds][colliding_inds]

        offsets = 0.5 * torch.normal(0.05, 0.05, size=(points.shape[0] * n_upsample, 1), device=norms.device).abs()
        verts, norms = points.repeat(n_upsample, 1), norms.repeat(n_upsample, 1)
        points = torch.cat((points, (verts - offsets * norms).reshape(-1, 3)), dim=0)

    if points.shape[0] > max_queries:
        points = points[torch.randperm(points.size(0), device=points.device)[:max_queries]]

    return points.float().reshape(1, -1, 3)  # add batch dimension

def keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, criterion, jts_valid):
    """
    Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """

    gt_root = gt_keypoints_3d[:, :1, :]
    gt_keypoints_3d = gt_keypoints_3d - gt_root
    pred_root = pred_keypoints_3d[:, :1, :]
    pred_keypoints_3d = pred_keypoints_3d - pred_root

    return joints_loss(pred_keypoints_3d, gt_keypoints_3d, criterion, jts_valid)


def object_kp3d_loss(pred_3d, gt_3d, criterion, is_valid):
    num_kps = pred_3d.shape[1] // 2
    pred_3d_ra = subtract_root_batch(pred_3d, root_idx=num_kps)
    gt_3d_ra = subtract_root_batch(gt_3d, root_idx=num_kps)
    loss_kp = vector_loss(
        pred_3d_ra,
        gt_3d_ra,
        criterion=criterion,
        is_valid=is_valid,
    )
    return loss_kp


def hand_kp3d_loss(pred_3d, gt_3d, criterion, jts_valid):
    pred_3d_ra = subtract_root_batch(pred_3d, root_idx=0)
    gt_3d_ra = subtract_root_batch(gt_3d, root_idx=0)
    loss_kp = keypoint_3d_loss(
        pred_3d_ra, gt_3d_ra, criterion=criterion, jts_valid=jts_valid
    )
    return loss_kp


def vector_loss(pred_vector, gt_vector, criterion, is_valid=None):
    dist = criterion(pred_vector, gt_vector)
    if is_valid.sum() == 0:
        return torch.zeros((1)).to(gt_vector.device)
    if is_valid is not None:
        valid_idx = is_valid.long().bool()
        dist = dist[valid_idx]
    loss = dist.mean().view(-1)
    return loss


def joints_loss(pred_vector, gt_vector, criterion, jts_valid):
    dist = criterion(pred_vector, gt_vector)
    if jts_valid is not None:
        dist = dist * jts_valid[:, :, None]
    loss = dist.mean().view(-1)
    return loss


def mano_loss(pred_rotmat, pred_betas, gt_rotmat, gt_betas, criterion, is_valid=None):
    loss_regr_pose = vector_loss(pred_rotmat, gt_rotmat, criterion, is_valid)
    loss_regr_betas = vector_loss(pred_betas, gt_betas, criterion, is_valid)
    return loss_regr_pose, loss_regr_betas
