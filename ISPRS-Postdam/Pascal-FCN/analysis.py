
"""
Various MISC code for real-time analysis of the CNN network. This works along the real-time
signal handler included in the TRAINING script allowing real-time analysis of the state of the network 

"""

# ======================= BATCH Visualizations ======================= #

# View Image in particular batch
plt.show(plt.imshow(np.rollaxis(solver.net.blobs['image'].data[0, ::-1],0,3)))

# View Labels in particular batch
plt.show(plt.imshow(solver.net.blobs['label'].data[0,0], vmin=0, vmax=6, cmap='gist_ncar'))

# View Prediction of particular state in the network - result without softmax
plt.show(plt.imshow(solver.net.blobs['score'].data[0,:].argmax(0), vmin=0, vmax=6, cmap='gist_ncar'))

# View class-boundaries of particular batch
vis(solver.net.blobs['edge'].data[0,:])

# Visualize absolute difference between infered class-boudnaries and label-boundaries
vis(np.abs(solver.net.blobs['edges'].data[0,:]-solver.net.blobs['edge-label'].data[0,:]))

# Visualize difference between between infered annotation and label-annotation
plt.show(plt.imshow(np.abs(solver.net.blobs['score'].data[0,:].argmax(0)-solver.net.blobs['label'].data[0,0]), vmin=0, vmax=6, cmap='gist_ncar'))


# ================= ERROR GRAPHS  ================== #

# plot step-error of annotation through epochs and fit straight trending line
plt.show(plt.plot(range(len(np.array(batch_error))), np.poly1d(np.polyfit(range(len(np.array(batch_error))), np.array(batch_error),1))(range(len(np.array(batch_error)))), range(len(np.array(batch_error))), np.array(batch_error), '*'))                              
                                                                                           # plot step-error of edges through epochs and fit straight trending line                   
plt.show(plt.plot(range(len(np.array(edge_error))), np.poly1d(np.polyfit(range(len(np.array(edge_error))), np.array(edge_error),1))(range(len(np.array(edge_error)))), range(len(np.array(edge_error))), np.array(edge_error), '*'))                              
