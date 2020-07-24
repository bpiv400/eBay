import torch
from constants import VALIDATION
from rlenv.generate.Generator import DiscrimGenerator


def main():
    gen = DiscrimGenerator(verbose=True)
    gen.process_chunk(part=VALIDATION, chunk=0)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    main()
