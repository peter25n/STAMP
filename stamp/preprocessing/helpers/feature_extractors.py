import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import PIL
import numpy as np
#no marugoto dependency
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
import json
import h5py
import uni
import os
import timm

from .swin_transformer import swin_tiny_patch4_window7_224, ConvStem

__version__ = "001_01-10-2023"

def get_digest(file: str):
    sha256 = hashlib.sha256()
    with open(file, 'rb') as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

class FeatureExtractorCTP:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles.
        """
        digest = get_digest(self.checkpoint_path)
        assert digest == '7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539'

        self.model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
        self.model.head = nn.Identity()

        ctranspath = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ctranspath['model'], strict=True)
        
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model_name='xiyuewang-ctranspath-7c998680'

        print("CTransPath model successfully initialised...\n")
        return model_name
        
class FeatureExtractorUNI:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles. 
        Requirements: 
            Permission from authors via huggingface: https://huggingface.co/MahmoodLab/UNI
            Huggingface account with valid login token
        On first model initialization, you will be prompted to enter your login token. The token is
        then stored in ./home/<user>/.cache/huggingface/token. Subsequent inits do not require you to re-enter the token. 

        Args:
            device: "cuda" or "cpu"
        """
        asset_dir = f"{os.environ['STAMP_RESOURCES_DIR']}/uni"
        model, transform = uni.get_encoder(enc_name="uni", device=device, assets_dir=asset_dir)
        self.model = model
        self.transform = transform

        digest = get_digest(f"{asset_dir}/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin")
        model_name = f"mahmood-uni-{digest[:8]}"

        print("UNI model successfully initialised...\n")
        return model_name
    

class FeatureExtractorProvGP:
    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles using GigaPath tile encoder."""

        assets_dir = f"{os.environ['STAMP_RESOURCES_DIR']}"
        model_name = 'prov-gigapath'
        checkpoint = 'pytorch_model.bin'
        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(assets_dir, model_name, checkpoint)

        # Ensure the directory exists
        os.makedirs(ckpt_dir, exist_ok=True)

        # Load the model structure
        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)

        # Load the state dict from the checkpoint file
        self.model.load_state_dict(torch.load(ckpt_path))

        # If a CUDA device is available, move the model to the device
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        # Define the transform
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # if not os.path.isfile(ckpt_path):
        #     from huggingface_hub import login, hf_hub_download
        #     login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens
        #     os.makedirs(ckpt_dir, exist_ok=True)
        #     hf_hub_download('prov-gigapath/prov-gigapath', filename=checkpoint, local_dir=ckpt_dir, force_download=True)

        # state_dict = torch.load(ckpt_path, map_location="cpu")
        # self.model.load_state_dict(state_dict, strict=True)
        
        print("GigaPath tile encoder model successfully initialized...\n")
        return model_name

class FeatureExtractorProvGPSlide:
    def init_feat_extractor(self, device: str, **kwargs):
        """Initializes the GigaPath tile and slide encoders."""
        assets_dir = f"{os.environ.get('STAMP_RESOURCES_DIR', './assets')}"
        model_name = 'prov-gigapathslide'
        checkpoint = 'pytorch_model.bin'
        slide_enc = 'slide_model.bin'
        ckpt_dir = os.path.join(assets_dir, model_name)
        ckpt_path = os.path.join(ckpt_dir, checkpoint)
        slide_enc_path = os.path.join(ckpt_dir, slide_enc)

        # Initialize and load tile encoder model
        self.model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False)
        tile_state_dict = torch.load(ckpt_path)
        self.model.load_state_dict(tile_state_dict)

        # Add the provgp directory to the Python path
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'provgp')))
        # Initialize and load slide encoder model
        from gigapath import slide_encoder
        self.slide_encoder = slide_encoder.create_model(slide_enc_path, "gigapath_slide_enc12l768d", 1536)

        slide_state_dict = torch.load(slide_enc_path)
        self.slide_encoder.load_state_dict(slide_state_dict)
        
        if torch.cuda.is_available():
            self.model = self.model.to(device)
            self.slide_encoder = self.slide_encoder.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # Ensure the directory exists
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save the model
        torch.save(self.model.state_dict(), ckpt_path)
        torch.save(self.slide_encoder.state_dict(), slide_enc_path)

        print("GigaPath tile and slide encoder models successfully initialized...\n")
        return model_name

'''
class FeatureExtractorPhikon:
    class FeatureExtractorPhikon:
    def init_feat_extractor(self, device: str, weights_path: str = None, **kwargs):
        """Initializes the Phikon model."""
        # Load the pretrained Phikon model
        self.image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

        # Move the model to the device if CUDA is available
        if torch.cuda.is_available():
            self.model = self.model.to(device)
        
        model_name = 'phikon'
        
        if weights_path:
            self.save_model(weights_path)
        
        print("Phikon model successfully initialized...\n")
        return model_name

class FeatureExtractorRetCCL:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles.
        """
        digest = get_digest(self.checkpoint_path)
        assert digest == '931956f31d3f1a3f6047f3172b9e59ee3460d29f7c0c2bb219cbc8e9207795ff'

        # Initialize the RetCCL model
        self.model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        self.model.fc = nn.Identity()

        retccl = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(retccl, strict=True)
        
        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model_name = 'retccl-model'

        print("RetCCL model successfully initialized...\n")
        return model_name


class FeatureExtractorHIPT:
    def init_feat_extractor(self, device: str, **kwargs):
        # Initialize the HIPT model
        # Example initialization logic
        self.model = ...  # Replace with actual model initialization
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model_name = 'hipt-model'
        print("HIPT model successfully initialized...\n")
        return model_name


class FeatureExtractorLunit:
    def __init__(self, model_key: str = "DINO_p16"):
        self.model_key = model_key

    def init_feat_extractor(self, device: str, **kwargs):
        """Extracts features from slide tiles."""
        # Initialize the Lunit model
        self.model = vit_small(pretrained=True, progress=True, key=self.model_key)

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model_name = 'lunit-dino-p16-model'

        print("Lunit DINO model successfully initialized...\n")
        return model_name

class FeatureExtractorRemedis:
    def init_feat_extractor(self, device: str, **kwargs):
        # Initialize the Remedis model
        # Example initialization logic
        self.model = ...  # Replace with actual model initialization
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model_name = 'remedis-model'
        print("Remedis model successfully initialized...\n")
        return model_name

class FeatureExtractorPathoDuet:
    def init_feat_extractor(self, device: str, **kwargs):
        # Initialize the PathoDuet model
        # Example initialization logic
        self.model = ...  # Replace with actual model initialization
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model_name = 'pathoduet-model'
        print("PathoDuet model successfully initialized...\n")
        return model_name

class FeatureExtractorBEPH:
    def init_feat_extractor(self, device: str, **kwargs):
        # Initialize the BEPH model
        # Example initialization logic
        self.model = ...  # Replace with actual model initialization
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model_name = 'beph-model'
        print("BEPH model successfully initialized...\n")
        return model_name

class FeatureExtractorCONCH:
    def init_feat_extractor(self, device: str, **kwargs):
        # Initialize the CONCH model
        # Example initialization logic
        self.model = ...  # Replace with actual model initialization
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model_name = 'conch-model'
        print("CONCH model successfully initialized...\n")
        return model_name

class FeatureExtractorCiga:
    def init_feat_extractor(self, device: str, **kwargs):
        # Initialize the Ciga model
        # Example initialization logic
        self.model = ...  # Replace with actual model initialization
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        model_name = 'ciga-model'
        print("Ciga model successfully initialized...\n")
        return model_name


'''



class SlideTileDataset(Dataset):
    def __init__(self, patches: np.array, transform=None, *, repetitions: int = 1) -> None:
        self.tiles = patches
        #assert self.tiles, f'no tiles found in {slide_dir}'
        self.tiles *= repetitions
        self.transform = transform

    # patchify returns a NumPy array with shape (n_rows, n_cols, 1, H, W, N), if image is N-channels.
    # H W N is Height Width N-channels of the extracted patch
    # n_rows is the number of patches for each column and n_cols is the number of patches for each row
    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        image = PIL.Image.fromarray(self.tiles[i])
        if self.transform:
            image = self.transform(image)

        return image

def extract_features_(
        *,
        model, model_name, transform, norm_wsi_img: np.ndarray, coords: list, wsi_name: str, outdir: Path,
        augmented_repetitions: int = 0, cores: int = 8, is_norm: bool = True, device: str = 'cpu',
        target_microns: int = 256, patch_size: int = 224, slide_encoder=None,
) -> None:
    """Extracts features from slide tiles.

    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """

    # Obsolete (?)
    # augmenting_transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
    #     transforms.RandomHorizontalFlip(p=.5),
    #     transforms.RandomVerticalFlip(p=.5),
    #     transforms.RandomApply([transforms.GaussianBlur(3)], p=.5),
    #     transforms.RandomApply([transforms.ColorJitter(
    #         brightness=.1, contrast=.2, saturation=.25, hue=.125)], p=.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    extractor_string = f'STAMP-extract-{__version__}_{model_name}'
    with open(outdir.parent/'info.json', 'w') as f:
        json.dump({'extractor': extractor_string,
                  'augmented_repetitions': augmented_repetitions,
                  'normalized': is_norm,
                  'microns': target_microns,
                  'patch_size': patch_size}, f)

    unaugmented_ds = SlideTileDataset(norm_wsi_img, transform)
    augmented_ds = []

    #clean up memory
    del norm_wsi_img

    ds = ConcatDataset([unaugmented_ds, augmented_ds])
    dl = torch.utils.data.DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=cores, drop_last=False, pin_memory=device != 'cpu')

    model = model.eval().to(device)
    dtype = next(model.parameters()).dtype

    feats = []
    with torch.inference_mode():
        for batch in tqdm(dl, leave=False):
            feats.append(
                model(batch.type(dtype).to(device)).half().cpu().detach())

        all_feats = torch.concat(feats)

    if slide_encoder:
        slide_encoder.eval().to(device)
        with torch.inference_mode():
            all_feats = slide_encoder(all_feats.to(device))
        all_feats = all_feats.cpu().detach()

    with h5py.File(f'{outdir}.h5', 'w') as f:
        f['coords'] = coords
        f['feats'] = all_feats.cpu().numpy()
        f['augmented'] = np.repeat(
            [False, True], [len(unaugmented_ds), len(augmented_ds)])
        assert len(f['feats']) == len(f['augmented'])
        f.attrs['extractor'] = extractor_string