import os
import glob
import torch
import argparse
import pywavefront
import numpy as np
from numpy.random import default_rng
from igl import winding_number


def read_obj(file_path):
    obj = pywavefront.Wavefront(file_path, collect_faces=True, parse=True)
    verts = obj.vertices
    faces = []
    for mesh_name in obj.meshes:
        part = obj.meshes[mesh_name]
        faces += part.faces

    return verts, faces

def normalize_shape(verts):
        # center
        c = np.mean(verts, axis=0)
        verts -= c
        # scale
        s = 0.4 / np.abs(verts).max()
        verts *= s
        # translate to [0,1]
        verts += np.array([.5, .5, .5])

        return verts

def create_grid(grid_res):
    d = grid_res
    num_points = grid_res ** 3

    # d = round(num_points ** (1/3))
    # self.grid_res = d
    shape = (d, d, d)
    pzs = torch.linspace(0, 1, d)
    pys = torch.linspace(0, 1, d)
    pxs = torch.linspace(0, 1, d)
    pzs = pzs.view(-1, 1, 1).expand(*shape).contiguous().view(num_points)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(num_points)
    pxs = pxs.view(1, 1, -1).expand(*shape).contiguous().view(num_points)
    pts = torch.stack([pzs, pys, pxs], dim=1).numpy()

    return pts

def _sample_points(grid_pts, case=8, grid_res=32):
        rng = default_rng()
        num_points = grid_res ** 3
        o = 1. / (2 * (grid_res - 1.))
        if case == 0:
            pts = grid_pts
        elif case == 1:
            pts = grid_pts + np.array([[o ,o, o]])
        elif case == 2:
            pts = grid_pts + np.array([[0 ,o, o]]) 
        elif case == 3:
            pts = grid_pts + np.array([[o ,0, o]])
        elif case == 4:
            pts = grid_pts + np.array([[0 ,0, o]])
        elif case == 5:
            pts = grid_pts + np.array([[o ,o, 0]]) 
        elif case == 6:
            pts = grid_pts + np.array([[0 ,o, 0]])
        elif case == 7:
            pts = grid_pts + np.array([[o ,0, 0]])
        elif case == 8:
            pts = grid_pts + (rng.uniform(size=(num_points, 3)) / (grid_res - 1.))
        pts = np.clip(pts, 0., 1.) # TODO: change to add offsets between [-1/res, 1/res] but need to change the coord2index method to support negative numbers

        return pts

def get_occ(source_file, grid_pts, grid_res, case):
    name = os.path.split(source_file)[1][:-4]

    verts, faces = read_obj(source_file)
    verts, faces = np.array(verts), np.array(faces)
    verts = normalize_shape(verts)

    # sample points
    pts = _sample_points(grid_pts, case=case, grid_res=grid_res)        
    
    w_num = winding_number(verts, faces, pts)
    occ_val = torch.tensor(w_num).round()
    occ_val = occ_val.reshape(1, pts.shape[0]).float()
    if not torch.all(torch.logical_or(occ_val == 1, occ_val == 0)):
        print(f'Winding numbers not all 0 or 1 for {name}')
        occ_val[occ_val < 0] = 0.
        occ_val[occ_val > 1] = 1.

    return occ_val, torch.tensor(pts).float()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Convert')

    parser.add_argument("--cat", type=str, choices=['chair', 'car', 'rifle', 'table', 'airplane'])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--grid_res", type=int, default=32)

    args, unknown = parser.parse_known_args()

    base_dir = '/net/projects/ranalab/meitars/diffusion-3d/data'
    data_dir = os.path.join(base_dir, args.cat + '_data')
    data_dir = os.path.join(data_dir, glob.glob(data_dir + '/*')[0])
    will_data_dir = '/net/projects/ranalab/aprilwang/shapenet/chairs_500'

    sorted_files_dir = sorted(glob.glob(data_dir + '/*'))    
    will_sorted_files_dir = sorted(glob.glob(will_data_dir + '/*'))    
    num_files = len(sorted_files_dir)

    grid_pts = create_grid(args.grid_res)

    if args.end_idx > num_files or args.end_idx == -1:
        args.end_idx = num_files

    for file_idx in range(args.start_idx, args.end_idx):
        # input_file = os.path.join(sorted_files_dir[file_idx], 'watertight_model.obj')
        input_file = os.path.join(sorted_files_dir[file_idx], 'model.obj')
        will_input_file = will_sorted_files_dir[file_idx]
        print(f'start with number {file_idx}/{args.end_idx}')

        # if os.path.isfile(input_file.replace('manifold_model.obj', 'occupacies_29.pt')):
        #     print(f'{file_idx} exists')
        # else:
        #     print(f'{file_idx} DOES NOT exists')

        if not os.path.isfile(input_file):
            print(f'file {input_file} (idx:{file_idx}) DOES NOT EXISTS!!. skipping.')
            continue
        if not os.path.isfile(will_input_file):
            print(f'file (will) {will_input_file} (idx:{file_idx}) DOES NOT EXISTS!!. skipping.')
            continue
        
        for sample in range(30):
            case = sample if sample <= 8 else 8
            # output_path_occ = input_file.replace('watertight_model.obj', f'will_occupacies_{sample}.pt')
            # output_path_pts = input_file.replace('watertight_model.obj', f'will_pts_{sample}.pt')
            output_path_occ = input_file.replace('model.obj', f'will_occupacies_{sample}.pt')
            output_path_pts = input_file.replace('model.obj', f'will_pts_{sample}.pt')
            if os.path.isfile(output_path_occ):
                # print(f'skipping {output_path_occ} as it is already exists')
                continue
            # occs, pts = get_occ(input_file, grid_pts, args.grid_res, case=case)
            occs, pts = get_occ(will_input_file, grid_pts, args.grid_res, case=case)
            # write to output_path
            torch.save(occs, output_path_occ)
            torch.save(pts, output_path_pts)
        print(f'Done with number {file_idx}/{args.end_idx}')

    print("Done")
