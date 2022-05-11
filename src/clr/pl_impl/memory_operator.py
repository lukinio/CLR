from typing import Union
import torch


class MemoryOperator:

    def __init__(self, memory_length: int):
        self.__memory_length = memory_length
        self.__memory: Union[None, torch.Tensor] = None

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        if self.__memory is None:
            latent_to_return = latent
        else:
            latent_to_return = torch.cat((latent, self.__memory))

        self.__memory = latent_to_return[0:self.__memory_length].detach()
        return latent_to_return
