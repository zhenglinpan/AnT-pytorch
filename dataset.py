from torch.utils.data import Dataset

class AnimationDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def __getitem__(self, index):
        """
            __getitem__ should return the bboxes coordinates (x, y, w, h) of M 
            ocluded segments found in an image and their respective color id.
            And also, resized patches of ocluded segments
            :patches para: size([M, 32, 32])
            :bboxes_info para: size([M, 4])
            :color_ids para: size([M, 1])
        """
        return {'patches_ref': patches_ref, 
                'patches_target': patches_target,
                'info_ref': [[bbox_info_ref]] + [[color_ids_ref]],
                'info_target': [[bbox_info_target]] + [[color_ids_target]]
                }
    
    def __len__(self, ):
        pass