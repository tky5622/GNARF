def _complex_to_rgb(array):
    from matplotlib.colors import hsv_to_rgb
    hue = (np.angle(array)+np.pi)/np.pi/2
    sat = np.abs(array)
    sat = (sat/np.max(sat))
    hsv = np.stack([hue, sat, np.full_like(hue, 1.0)], axis=-1)

    rgb = hsv_to_rgb(hsv)
    return (rgb * 255).astype('uint8')

def _add_sdf_contour(img, vertices, faces, points_grid, plane='xy'):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    height, width = points_grid.shape[:2]
    fig = Figure(figsize=(height/300, width/300), dpi=300)
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    ax.axis('off')
    fig.tight_layout(pad=0)
    # To remove the huge white borders
    ax.margins(0)
    # TODO
    ax.imshow(img, origin='lower')
    # ax.imshow(img, origin, extent)
    sdf_f = pysdf.SDF(vertices, faces.astype('uint32'))
    shp = points_grid.shape[:-1]
    sdf_grid = sdf_f(points_grid.reshape(-1, 3)).reshape(shp)
    if plane == 'xy':
        xy = points_grid[:,:,[0,1]]
    elif plane == 'xz':
        xy = points_grid[:,:,[0,2]]
    elif plane == 'yz':
        xy = points_grid[:,:,[1,2]]
    else:
        raise ValueError
    xy[:,:,0] = (xy[:,:,0] - xy[:,:,0].min())/(xy[:,:,0].max() - xy[:,:,0].min())*(width -1)
    xy[:,:,1] = (xy[:,:,1] - xy[:,:,1].min())/(xy[:,:,1].max() - xy[:,:,1].min())*(height-1)
    ax.contour(xy[:,:,0], xy[:,:,1], sdf_grid, levels=(0.0,), alpha=1.0, colors='k', origin='lower')
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape((height, width, -1))
    return image_from_plot
