import odl
import numpy as np

def GeoParaSetting(sampleNum=720,detector_num=720):
    params = {}
    
    params['N'] = 512
    params['M'] = 512
    params['img_pixel_size_x'] = 0.15#0.5 # mm
    params['img_pixel_size_y'] = 0.15#0.5 # mm
    # Number of detector pixels
    params['num_detector_pixels'] = detector_num
    params['det_pixel_size'] = 0.2 # mm
    
    # Number of projection angles.
    params['num_angles'] = sampleNum
    params['source_origin'] = 950#2240.0
    
    params['detector_origin'] = 200#2240.0
    
    # Filter Type
    params['filter_type'] = 'Ram-Lak'
    params['frequency_scaling'] = 1
    return params


def FanFlatFBP(sinogram, params, isFilter=True):
    reco_space = odl.uniform_discr(
    min_pt=[-params['N']*params['img_pixel_size_x']//2, 
            -params['N']*params['img_pixel_size_x']//2], 
    max_pt=[params['M']*params['img_pixel_size_y']//2, 
            params['M']*params['img_pixel_size_y']//2], 
    shape=[params['N'], params['M']], dtype='float32')

    grid = odl.uniform_grid(0, 2*np.pi, params['num_angles'])

    angle_partition = odl.uniform_partition_fromgrid(grid)
    detector_partition = odl.uniform_partition(-params['num_detector_pixels']*
                                               params['det_pixel_size']//2, 
                                               params['num_detector_pixels']*
                                               params['det_pixel_size']//2, 
                                               params['num_detector_pixels'])
    
    geometry = odl.tomo.FanBeamGeometry(
        angle_partition, detector_partition, 
        src_radius=params['source_origin'], 
        det_radius=params['detector_origin'])

    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
    # img = np.load('/dev/shm/train_img/1.npy')
    # phantom = reco_space.element(img)
    # proj_data = ray_trafo(phantom)

    if isFilter:
        fbp = odl.tomo.fbp_op(ray_trafo, 
                              filter_type=params['filter_type'],
                              frequency_scaling=params['frequency_scaling'])
        
        fbp_reconstruction = fbp(sinogram)
    else:
        fbp_reconstruction = ray_trafo.adjoint(sinogram)
        
    rec_fbp = fbp_reconstruction.data
    rec_fbp[rec_fbp<0] = 0
    return rec_fbp


def FanFlatTransform(img, params):
    reco_space = odl.uniform_discr(
    min_pt=[-params['N']*params['img_pixel_size_x']//2, 
            -params['N']*params['img_pixel_size_x']//2], 
    max_pt=[params['M']*params['img_pixel_size_y']//2, 
            params['M']*params['img_pixel_size_y']//2], 
    shape=[params['N'], params['M']], dtype='float32')

    grid = odl.uniform_grid(0, 2*np.pi, params['num_angles'])

    angle_partition = odl.uniform_partition_fromgrid(grid)
    detector_partition = odl.uniform_partition(-params['num_detector_pixels']*
                                               params['det_pixel_size']//2, 
                                               params['num_detector_pixels']*
                                               params['det_pixel_size']//2, 
                                               params['num_detector_pixels'])

    geometry = odl.tomo.FanBeamGeometry(
        angle_partition, detector_partition, 
        src_radius=params['source_origin'], 
        det_radius=params['detector_origin'])

    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
    phantom = reco_space.element(img)
    proj_data = ray_trafo(phantom).data

    return proj_data