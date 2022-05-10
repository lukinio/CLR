import numpy as np

from pl_bolts.utils import _OPENCV_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.transforms import (Compose, ColorJitter, RandomResizedCrop, RandomHorizontalFlip, RandomApply,
                                        RandomGrayscale, Resize, CenterCrop, ToTensor)
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

if _OPENCV_AVAILABLE:
    import cv2
else:  # pragma: no cover
    warn_missing_pkg("cv2", pypi_name="opencv-python")


class CLRTrainDataTransform:
    """Transforms for CLR.

    Transform::

        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1.0, normalize=None,
            augment_both: bool = True
    ) -> None:

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize
        self.augment_both = augment_both

        self.color_jitter = ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        data_transforms = [
            RandomResizedCrop(size=self.input_height),
            RandomHorizontalFlip(p=0.5),
            RandomApply([self.color_jitter], p=0.8),
            RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(GaussianBlur(kernel_size=kernel_size, p=0.5))

        data_transforms = Compose(data_transforms)

        if normalize is None:
            self.final_transform = ToTensor()
        else:
            self.final_transform = Compose([ToTensor(), normalize])

        self.train_transform = Compose([data_transforms, self.final_transform])

        # add online train transform of the size of global view
        self.online_transform = Compose(
            [RandomResizedCrop(self.input_height), RandomHorizontalFlip(p=0.5), self.final_transform]
        )

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample) if self.augment_both else self.final_transform(sample)
        xj = transform(sample)

        return xi, xj, self.online_transform(sample)


class CLREvalDataTransform(CLRTrainDataTransform):
    """Transforms for CLR.

    Transform::

        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1.0, normalize=None
    ):
        super().__init__(
            normalize=normalize, input_height=input_height, gaussian_blur=gaussian_blur, jitter_strength=jitter_strength
        )

        # replace online transform with eval time transform
        self.online_transform = Compose(
            [
                Resize(int(self.input_height + 0.1 * self.input_height)),
                CenterCrop(self.input_height),
                self.final_transform,
            ]
        )


class CLRFinetuneTransform:
    def __init__(
        self, input_height: int = 224, jitter_strength: float = 1.0, normalize=None, eval_transform: bool = False
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.normalize = normalize

        self.color_jitter = ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        if not eval_transform:
            data_transforms = [
                RandomResizedCrop(size=self.input_height),
                RandomHorizontalFlip(p=0.5),
                RandomApply([self.color_jitter], p=0.8),
                RandomGrayscale(p=0.2),
            ]
        else:
            data_transforms = [
                Resize(int(self.input_height + 0.1 * self.input_height)),
                CenterCrop(self.input_height),
            ]

        if normalize is None:
            final_transform = ToTensor()
        else:
            final_transform = Compose([ToTensor(), normalize])

        data_transforms.append(final_transform)
        self.transform = Compose(data_transforms)

    def __call__(self, sample):
        return self.transform(sample)


class GaussianBlur:
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `GaussianBlur` from `cv2` which is not installed yet.")

        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample
