
# Code adapted from https://github.com/noefford/Image-Adaptive-YOLO-pytorch
import torch
def rgb2lum(image):
  image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :,
                                                  1] + 0.06 * image[:, :, :, 2]
  return image[:, :, :, None]


def tanh01(x):
  return torch.tanh(x) * 0.5 + 0.5


def tanh_range(l, r, initial=None):

  def get_activation(left, right, initial):

    def activation(x):
      if initial is not None:
        bias = torch.atanh(2 * (initial - left) / (right - left) - 1)
      else:
        bias = 0
      return tanh01(x + bias) * (right - left) + left

    return activation

  return get_activation(l, r, initial)


def lerp(a, b, l):
  return (1 - l) * a + l * b

