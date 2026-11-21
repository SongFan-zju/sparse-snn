"""Activation sites of the CIFAR ResNet18 backbone."""

def activation_sites(source):
    yield 'relu', source.bn1.num_features
    for stage in range(1,getattr(source,'num_stages',4)+1):
        for index,block in enumerate(getattr(source,f'layer{stage}')):
            for k in (1,2):
                yield f'layer{stage}.{index}.relu{k}',getattr(block,f'bn{k}').num_features
