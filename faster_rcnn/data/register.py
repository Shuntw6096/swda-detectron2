from detectron2.data.datasets import register_pascal_voc
from pathlib import Path

dataset_base_dir = Path(__file__).parent.parent.parent / 'datasets'



dataset_dir = str(dataset_base_dir/ 'itri-taiwan-416-VOCdevkit2007')
classes = ('person', 'two-wheels', 'four-wheels')
years = 2007
split = 'train' # "train", "test", "val", "trainval"
meta_name = 'itri-taiwan-416_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)


dataset_dir = str(dataset_base_dir/ 'tokyo-320-v2-VOCdevkit2007')
classes = ('person', 'two-wheels', 'four-wheels')
split = 'train' # "train", "test", "val", "trainval"
meta_name = 'tokyo-320-v2_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)


dataset_dir = str(dataset_base_dir/ 'tokyo-320-test-only-VOCdevkit2007')
split = 'test' # "train", "test", "val", "trainval"
classes = ('person', 'two-wheels', 'four-wheels')
years = 2007
meta_name = 'tokyo-320_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)


dataset_dir = str(dataset_base_dir/ 'tokyo-320-v2-tuning-VOCdevkit2007')
split = 'train' # "train", "test", "val", "trainval"
classes = ('person', 'two-wheels', 'four-wheels')
years = 2007
meta_name = 'tokyo-320-v2-tuning_{}'.format(split)
register_pascal_voc(meta_name, dataset_dir, split, years, classes)