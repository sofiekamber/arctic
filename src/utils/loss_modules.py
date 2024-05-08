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
    coap_loss_r = coap_loss(pred["object.v.cam"], pred["mano.beta.r"], pred["mano.joints3d.r"], is_right=True)
    coap_loss_l = coap_loss(pred["object.v.cam"], pred["mano.beta.l"], pred["mano.joints3d.l"], is_right=False)

    return coap_loss_r, coap_loss_l


def coap_loss(pred_v3d_object, pred_betas_r, pred_joints3d_r, is_right):
    batch_size = pred_v3d_object.shape[0]
    model = build_mano_coap(is_right, batch_size)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = coap.attach_coap(model, pretrained=True, device=device) #Need model with pca


    #prep data
    torch_param = {}
    torch_param['hand_pose'] = pred_joints3d_r[:, :16,:].view(batch_size, 48) #crop 21 joints to 16
    torch_param['betas'] = pred_betas_r
    #torch_param['transl'] = torch.from_numpy(np.array([[-5.0101800e-003, 1.5031957e-001, 3.0754370e-002]])).to(torch.float32).to(args.device) #TODO:?


    mano_output = model(**torch_param, return_verts=True, return_full_pose=True)
    assert model.joint_mapper is None, 'COAP requires valid SMPL joints as input'

    scene_points = sample_scene_points(mano_output, pred_v3d_object, device) #check only points inside Mano bbox
    if scene_points is None:
        return torch.zeros(8, device=device)

    coap_loss, _collision_mask = model.coap.collision_loss(scene_points, mano_output, ret_collision_mask=True)

    return coap_loss


@torch.no_grad()
def sample_scene_points(mano_output, scene_vertices, device, max_queries=10000):

    # remove points that are outside the Mano bounding box
    bb_min = mano_output.vertices.min(1).values
    bb_max = mano_output.vertices.max(1).values

    bb_min_expanded = bb_min.unsqueeze(1).expand_as(scene_vertices)
    bb_max_expanded = bb_max.unsqueeze(1).expand_as(scene_vertices)

    #mask containing all object points inside mano bbox
    mask = (scene_vertices >= bb_min_expanded).all(dim=-1) & (scene_vertices <= bb_max_expanded).all(dim=-1)

    max_true_count = min(mask.sum(dim=1).max(), max_queries)

    if max_true_count == 0:
        return None

    padded_scene_vertices = torch.empty((mask.shape[0], max_true_count, 3), dtype=scene_vertices.dtype, device=device)

    for i in range(scene_vertices.shape[0]):

        true_vertices = scene_vertices[i][mask[i]]

        # If the number of true vertices is less than max_true_count, pad with random vertices
        if true_vertices.size(0) < max_true_count:
            num_to_pad = max_true_count - true_vertices.size(0)

            # Randomly sample indices from the current batch
            random_indices = torch.randint(0, scene_vertices.shape[1], (num_to_pad,))

            random_vertices = scene_vertices[i][random_indices]

            padded_scene_vertices[i] = torch.cat((true_vertices, random_vertices), dim=0)
        else:
            padded_scene_vertices[i] = true_vertices[:max_true_count]

    return padded_scene_vertices.float()

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
