import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

depth = np.load('imgs/straight10_cropped.npy')
img = mpimg.imread('imgs/straight10_cropped.png')[:,:,:-1]
table_depth = 0.66
depth_post = np.where(depth > table_depth, 0, depth)
H, W = depth_post.shape

mold_diameter, eps = 1.15e-2, 0.05e-2 
# mold_diameter, eps = 1.1-2, 0.1e-2 

window_size = 20 # pixels on either side of current pixel
plot_proposals_fg_x, plot_proposals_fg_y, plot_proposals_fg_z = [], [], []
proposals_fg = []

plot_proposals_bg_x, plot_proposals_bg_y, plot_proposals_bg_z = [], [], []
proposals_bg = []

min_num_proposals = 2

for j in range(W):
    window_Wmin = 0 if j-window_size<0 else j-window_size
    window_Wmax = W-1 if j+window_size>=W else j+window_size
    for i in range(H):
        curr_pixel = depth_post[i,j]
        window_Hmin = 0 if i-window_size<0 else i-window_size
        window_Hmax = H-1 if i+window_size>=H else i+window_size

        propose_fg = False
        num_proposals_fg = 0
        max_diff = 0

        propose_bg = False
        num_proposals_bg = 0
        # combos = [[window_Hmin,j], [window_Hmax, j], [i,window_Wmin], [i,window_Wmax], [window_Hmin, window_Wmin], 
        #         [window_Hmin, window_Wmax], [window_Hmax, window_Wmin], [window_Hmax, window_Wmax]]
        combos = [[window_Hmin,j], [window_Hmax, j], [i,window_Wmin], [i,window_Wmax]]

        for combo in combos:
            m,n = combo
            diff = depth_post[m,n] - curr_pixel
            propose_fg = diff > mold_diameter-eps and diff < mold_diameter+eps
            if propose_fg:
                max_diff = diff if diff > max_diff else max_diff
                num_proposals_fg += 1
            
            diff_bg = curr_pixel - depth_post[m,n]
            propose_bg = diff_bg > mold_diameter-eps and diff_bg < mold_diameter+eps
            if propose_bg:
                num_proposals_bg += 1

        if propose_fg and num_proposals_fg >= min_num_proposals:
            plot_proposals_fg_x.append(j)
            plot_proposals_fg_y.append(i)
            plot_proposals_fg_z.append(num_proposals_fg)
            # proposals_fg.append(str((j,i)))
            proposals_fg.append([j,i])
            # proposals_fg[str((j,i))] = str(max_diff)
        
        if propose_bg and num_proposals_bg >= min_num_proposals:
            plot_proposals_bg_x.append(j)
            plot_proposals_bg_y.append(i)
            plot_proposals_bg_z.append(num_proposals_bg)
            # proposals.append(str((j,i)))
            proposals_bg.append([j,i])
            # proposals[str((j,i))] = str(max_diff)


# all_proposals = proposals_fg + proposals_bg
# # rgb_values = [img[proposal[1], proposal[0]] for proposal in all_proposals]
# depth_values = [depth_post[proposal[1], proposal[0]] for proposal in all_proposals]
# all_labels = len(proposals_fg) * [1] + len(proposals_bg) * [0]
# # final_proposals = {'fg': [], 'bg': []}
# point_labels = []

# plot_proposals_fg_x, plot_proposals_fg_y = [], []
# plot_proposals_bg_x, plot_proposals_bg_y = [], []

# neigh = NearestNeighbors(n_neighbors=10)
# # neigh.fit(all_proposals)
# neigh.fit(np.array(depth_values).reshape(-1,1))
# # neigh = KNeighborsClassifier(n_neighbors=5)
# # neigh.fit(all_proposals, all_labels)
# for depth_value, proposal in zip(depth_values, all_proposals):
#     # neighbor_inds = neigh.kneighbors(np.array([proposal]))[1][0]
#     neighbor_inds = neigh.kneighbors(np.array([[depth_value]]))[1][0]
#     tot_neighbor = neighbor_inds.shape[0]
#     if sum([all_labels[i] for i in neighbor_inds]) > tot_neighbor//2:
#         # final_proposals['fg'].append(proposal)
#         point_labels.append(1)
#         plot_proposals_fg_x.append(proposal[0])
#         plot_proposals_fg_y.append(proposal[1])
#     else:
#         # final_proposals['bg'].append(proposal)
#         point_labels.append(0)
#         plot_proposals_bg_x.append(proposal[0])
#         plot_proposals_bg_y.append(proposal[1])



# img = mpimg.imread('imgs/straight10_cropped.png')
# plt.imshow(img)
# plt.scatter(plot_proposals_x, plot_proposals_y, c=plot_proposals_z)
# plt.colorbar()
# plt.show()

depth_fg = [depth_post[proposal[1], proposal[0]] for proposal in proposals_fg]
depth_bg = [depth_post[proposal[1], proposal[0]] for proposal in proposals_bg]

# depth_fg = [max(depth_fg)-d for d in depth_fg]
# depth_bg = [max(depth_bg)-d for d in depth_bg]

plt.imshow(img)
plt.scatter(plot_proposals_fg_x, plot_proposals_fg_y, c=depth_fg)
plt.scatter(plot_proposals_bg_x, plot_proposals_bg_y, c=depth_bg)
plt.colorbar()
plt.show()

# the json file where the output must be stored
out_file = open("sam_proposals.json", "w")
json.dump({'point_coords': all_proposals, 'point_labels': point_labels}, out_file, indent = 6)
out_file.close()


# # the json file where the output must be stored
# out_file = open("sam_proposals.json", "w")
# json.dump(proposals, out_file, indent = 6)
# out_file.close()




