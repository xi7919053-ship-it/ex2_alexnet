"""把模型集中放在这里，导入时会方便一些。"""

from models.alexnet import AlexNet


# 根据传入的名字创建对应模型
def model_factory(model_name, num_classes=10):
    # 例如：model_factory('resnet18', 10)
    model_name = model_name.lower()
    if model_name == 'resnet18':
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError('Invalid model name: {}，only resnet18 is available'.format(model_name))
