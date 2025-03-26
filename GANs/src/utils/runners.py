import torch
from torch import optim
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
