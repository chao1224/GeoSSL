from Geom3D.datasets import Molecule3D
import gdown


if __name__ == "__main__":
    # url = 'https://drive.google.com/uc?id=' + "1C_KRf8mX-gxny7kL9ACNCEV4ceu_fUGy"
    # output = '../data/Molecule3D/data.zip'
    # gdown.download(url, output)

    root_dir = "../data/Molecule3D"
    split_mode = "random"
    test_dataset = Molecule3D(root=root_dir, transform=None, split='train', split_mode=split_mode)
