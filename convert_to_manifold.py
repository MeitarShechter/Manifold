import os
import glob
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Convert')

    parser.add_argument("--cat", type=str, choices=['chair', 'car', 'rifle', 'table', 'airplane'])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    args, unknown = parser.parse_known_args()

    base_dir = '/net/projects/ranalab/meitars/diffusion-3d/data'
    data_dir = os.path.join(base_dir, args.cat + '_data')
    data_dir = os.path.join(data_dir, glob.glob(data_dir + '/*')[0])

    sorted_files_dir = sorted(glob.glob(data_dir + '/*'))    
    num_files = len(sorted_files_dir)

    if args.end_idx > num_files or args.end_idx == -1:
        args.end_idx = num_files

    for file_idx in range(args.start_idx, args.end_idx):
        input_file = os.path.join(sorted_files_dir[file_idx], 'model.obj')
        output_path = input_file.replace('model.obj', 'watertight_model.obj')
        if os.path.isfile(output_path):
            print(f'skipping {output_path} as it is already exists')
            continue
        cmd = './build/manifold {} {}'.format(input_file, output_path)
        os.system(cmd)

    print("Done")