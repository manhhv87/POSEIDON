import os

from generation.coco_instance_generator import COCOInstanceGenerator
from extraction.coco_instance_extractor import COCOInstanceExtractor
from utils.normalization import COCONormalization
from utils.coco2yolo import COCO2YOLO

if __name__ == '__main__':

    # Set environment variables in Google Colab
    os.environ['POSEIDON_DATASET_PATH'] = '/content/SeaDronesSee/data'

    normalizator = COCONormalization()
    normalizator.normalize()

    # extractor = COCOInstanceExtractor()
    # extractor.dataset_stats()
    # extractor.extract('/content/SeaDronesSee/outputs')

    # generator = COCOInstanceGenerator()
    # generator.balance('/content/SeaDronesSee/outputs')

    # conversor = COCO2YOLO()
    # conversor.convert("/Users/pabloruizponce/Vainas/SDSYOLO", "SDSYOLO")

    # conversor = COCO2YOLO(augmented=True)
    # conversor.convert("/Users/pabloruizponce/Vainas/SDSYOLOAugmented", "SDSYOLOAugmented")
