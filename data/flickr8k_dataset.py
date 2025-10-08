import csv
from torch.utils.data import Dataset
from PIL import Image

class flickr8k_pretrain(Dataset):
    def __init__(self, ann_file, img_root, transform=None):
        """
        ann_file: đường dẫn tới captions.txt (CSV format: image,caption)
        img_root: thư mục chứa ảnh
        """
        self.transform = transform
        self.img_root = img_root
        self.samples = []

        # Đọc file CSV (có header "image,caption")
        with open(ann_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["image"].strip()
                caption = row["caption"].strip()
                self.samples.append({"image": img_name, "caption": caption})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(f"{self.img_root}/{sample['image']}").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        caption = sample["caption"]
        return image, caption
