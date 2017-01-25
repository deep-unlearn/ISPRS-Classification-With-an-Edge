
# View Image
plt.show(plt.imshow(np.rollaxis(solver.net.blobs['image'].data[0, ::-1],0,3)))

# View Labels
plt.show(plt.imshow(solver.net.blobs['label'].data[0,0], vmin=0, vmax=6, cmap='gist_ncar'))

# View Prediction
plt.show(plt.imshow(solver.net.blobs['score'].data[0,:].argmax(0), vmin=0, vmax=6, cmap='gist_ncar'))

# View Edges
vis(solver.net.blobs['edges'].data[0,:])

# Edge Difference
vis(np.abs(solver.net.blobs['edges'].data[0,:]-solver.net.blobs['edge-label'].data[0,:]))

# Image Difference
plt.show(plt.imshow(np.abs(solver.net.blobs['score'].data[0,:].argmax(0)-solver.net.blobs['label'].data[0,0]), vmin=0, vmax=6, cmap='gist_ncar'))



# ================= ERROR GRAPHS ================== #


# plot error through epochs and fit line
plt.show(plt.plot(range(len(np.array(batch_error))), np.poly1d(np.polyfit(range(len(np.array(batch_error))), np.array(batch_error),1))(range(len(np.array(batch_error)))), range(len(np.array(batch_error))), np.array(batch_error), '*'))                              
                                                                                                                  
