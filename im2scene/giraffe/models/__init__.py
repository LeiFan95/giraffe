from torch._C import device
import torch.nn as nn
from im2scene.giraffe.models import (
    decoder, generator, bounding_box_generator, neural_renderer, posenet, pose_encoder)

# Dictionaries
pose_encoder_dict = {
    'simple': pose_encoder.PoseEncoder,
}

decoder_dict = {
    'simple': decoder.Decoder,
}

generator_dict = {
    'simple': generator.Generator,
}

background_generator_dict = {
    'simple': decoder.Decoder,
}

bounding_box_generator_dict = {
    'simple': bounding_box_generator.BoundingBoxGenerator,
}

neural_renderer_dict = {
    'simple': neural_renderer.NeuralRenderer
}


class GIRAFFE(nn.Module):
    ''' GIRAFFE model class.

    Args:
        device (device): torch device
        discriminator (nn.Module): discriminator network
        generator (nn.Module): generator network
        generator_test (nn.Module): generator_test network
    '''

    def __init__(self, device=None, discriminator=None, generator=None, generator_test=None,
                 **kwargs):
        super().__init__()
        self.device = device

        self.discriminator = self.to_device(discriminator)
        self.generator = self.to_device(generator)
        self.generator_test = self.to_device(generator_test)

    def to_device(self, module):
        if module is not None:
            return module.to(self.device)
        else:
            return None

    def forward(self, batch_size, **kwargs):
        gen = self.generator_test
        if gen is None:
            gen = self.generator
        return gen(batch_size=batch_size)

    def generate_test_images(self):
        gen = self.generator_test
        if gen is None:
            gen = self.generator
        return gen()

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
