import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils.algo_utils import get_ind_path, get_stored_structure
from PIL import Image, ImageFilter, ImageDraw
from utils.algo_utils import best_in_exp

### VOXEL COLORS
EMPTY_VOXEL = (255.0/255.0, 255.0/255.0, 255.0/255.0, 0)
RIGID_VOXEL = (0.15, 0.15, 0.15)
SOFT_VOXEL = (0.75, 0.75, 0.75)
ACT_H_VOXEL = (0.99215, 0.55816, 0.24274)
ACT_V_VOXEL = (0.42768, 0.68678, 0.84095)
FIXED_VOXEL = (0.0, 0.0, 0.0)

### COLOR MAP
color_map = {   0: EMPTY_VOXEL,
                1: RIGID_VOXEL,
                2: SOFT_VOXEL,
                3: ACT_H_VOXEL,
                4: ACT_V_VOXEL,
                5: FIXED_VOXEL}


def render_body(body, file):
    """
    Save 'file' image of the body
    """
    # plot body
    fig, ax = plt.subplots()
    cmap = colors.LinearSegmentedColormap.from_list("", list(color_map.values()))
    boundaries = [0, 1, 2, 3, 4, 5, 6]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    ax.imshow(body, cmap=cmap, norm=norm)
    plt.axis('off')

    # save image
    plt.savefig(file, bbox_inches="tight", transparent=True)    # save image with no stroke
    add_stroke(file, 1) # override image, add stroke
    plt.close(fig)


def add_stroke(filename: str, size: int, color: str = 'black'):

    """     
    Code adapted from https://stackoverflow.com/a/70010492

    Add stroke to image.
    Args:
        filename:   name of the file on which generate the stroke (override)
        size:       size of the stroke
        color:      color og the stroke
    """
    img = Image.open(filename)
    X, Y = img.size
    edge = img.filter(ImageFilter.FIND_EDGES).load()
    stroke = Image.new(img.mode, img.size, (0,0,0,0))
    draw = ImageDraw.Draw(stroke)
    for x in range(X):
        for y in range(Y):
            if edge[x,y][3] > 0:
                draw.ellipse((x-size,y-size,x+size,y+size),fill=color)      # draw stroke
    stroke.paste(img, (0, 0), img )     # add stroke to image
    stroke.save(filename)



if __name__ == "__main__":

    gen_algo = False

    experiment_name = 'test_qd'
    store_in_order = True
    inds = best_in_exp(os.path.join('results', experiment_name), 6)    # render best in MAP-Elites exp
    generation = 29  # evogym only
    
    print()
    i=1
    for i in range(len(inds)):
        if gen_algo:
            result_base_dir = os.path.join('evogym', 'examples', 'saved_data') 
            structure_path = os.path.join(result_base_dir , experiment_name, 'generation_' + str(generation), 'structure', str(inds[i])+'.npz')
        else:
            result_base_dir = 'results' 
            structure_path = os.path.join(get_ind_path(inds[i], os.path.join(result_base_dir, experiment_name)), 'structure.npz')

        body = get_stored_structure(structure_path)[0]

        save_path = os.path.join(result_base_dir, experiment_name, 'images')
        try:
            os.makedirs(save_path)
        except:
            pass

        file_name = 'ind' + str(inds[i]) + '.png'
        if store_in_order:
            file_name = str(i) + '_' + file_name
        if gen_algo:
            file_name = 'gen' + str(generation) + '_ind' + str(inds[i]) + '.png'

        file_path = os.path.join(save_path, file_name)
        render_body(body, file_path)
        print("Saved image", file_name)
