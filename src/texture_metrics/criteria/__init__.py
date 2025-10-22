from .cnn import CNN, CNNOptions, RandomTripletDataset
from .fourier import (
    spectral_orthogonal_distance1D, 
    sliced_spectral_orthogonal_distance,
    spectral_radial_distance,
    sliced_spectral_radial_distance,
    radial_profile)
from .gradients import image_gradient
from .optimal_transport import (
    histogram_loss1D, 
    sliced_wasserstein_distance, 
    bure_distance)
from .slicing import sliced_distance
from .style_distances import (
    gramm, 
    weighted_mse_loss, 
    weighted_feature_distance
)