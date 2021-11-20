import torch
import GCL.augmentors as A
aug=A.Compose([A.EdgeAdding(pe=0.4)])
edge_index=torch.randint(0,11,(2,10))
x=torch.randn((10,128))
auged=aug(x,edge_index)
auged_x,auged_edge_index,_=auged