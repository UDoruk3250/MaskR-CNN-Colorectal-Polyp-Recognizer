import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import gdown

class DetectronPredictorEngine:
    def __init__(self):
        model_file_name = "polyp_model_final.pth"
        google_drive_link = "https://drive.google.com/uc?id=1Lm0AKnjBq2xfSOVhxn4CDEq4iyepmU_o"
        if not os.path.exists(model_file_name):
            print("Model file is downloading...")
            gdown.download(google_drive_link, model_file_name, quiet=True)
            print("Download complete!")
        
        self.cfg = get_cfg()
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        
        self.MODEL.WEIGHTS = model_file_name
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        
        self.predictor = DefaultPredictor(self.cfg)
        

    def __call__(self, image):
        return self.predictor(image)
