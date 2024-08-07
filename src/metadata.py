import torchvision.transforms as transforms

# subgroups of classes from ImageNet
CLASSES_DATASET = {
    'dogs': [232, 254, 151, 197, 178],
    'fungi': [992, 993, 994, 997],
    'birds': [10, 12, 14, 92, 95, 96],
    'cars': [468, 609, 627, 717, 779, 817],
    'datatest': [998, 999],
}

# normalization for ImageNet
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

VAL_TRANSFORM = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    NORMALIZE,
                ])

SGXL_MODEL = 'Imagenet-256'

NETWORK_URL = {
    "Imagenet-1024": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet1024.pkl",
    "Imagenet-512": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet512.pkl",
    "Imagenet-256": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl",
    "Imagenet-128": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet128.pkl",
    "Pokemon-512": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon512.pkl",
    "Pokemon-1024": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon1024.pkl",
    "Pokemon-256": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/pokemon256.pkl",
    "FFHQ-256": "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/ffhq256.pkl"
}

SEG_LABELS = {
    'dogs': ['background', 'leg', 'body', 'tail', 'tongue', 'eye', 'nose', 'snout', 'ear', 'head'],
    'fungi': ['background', 'cap', 'stem'],
    'birds': ['background', 'beak', 'eye', 'head', 'leg', 'wing', 'body', 'tail'],
    'cars': ['background', 'shell', 'tire', 'window'],
    'datatest': ['background', 'object1', 'object2'],
}