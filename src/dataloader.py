import os
import cv2
import numpy as np


class BOSSbaseLoader:
    def __init__(self, data_dir, subset_size=None):
        """
        BOSSbase veri setini yöneten sınıf.
        :param data_dir: .pgm dosyalarının bulunduğu dizin.
        :param subset_size: Geliştirme aşamasında sadece N adet görüntüyü okumak için.
        """
        self.data_dir = data_dir

        # Dizindeki tüm .pgm dosyalarını bul
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Dizin bulunamadı: {data_dir}. Lütfen BOSSbase verilerini kontrol edin."
            )

        self.image_paths = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".pgm")
            ]
        )

        # Test amaçlı alt küme seçimi
        if subset_size and subset_size < len(self.image_paths):
            self.iamge_paths = self.image_paths[:subset_size]

        print(
            f"Sistem Başlatıldı: Toplam {len(self.image_paths)} görüntü işleme alınacak."
        )

    def load_image(self, path):
        """
        Görüntüyü O(1) sabit zamanda belleğe gri tonlamalı 2D matris olarak yükler.
        """
        img_matrix = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_matrix is None:
            raise ValueError(f"Görüntü yüklenemedi: {path}. Dosya bozuk olabilir.")

        # BOSSbase standart boyut kontrolü
        if img_matrix.shape != (512, 512):
            print(
                f"Uyarı: {path} standart 512x512 boyutunda değil. Mevcut boyut: {img_matrix.shape}"
            )

        return img_matrix

    def calculate_base_metrics(self, img_matrix):
        """
        Görüntünün varyansını ve Shannon Entropisini hesaplar.
        """
        # 1. Küresel Varyans Hesaplaması
        variance = np.var(img_matrix)

        # 2. Shannon Entropisi Hesaplaması
        # Histogram çıkar (piksel frekansları)
        hist = cv2.calcHist([img_matrix], [0], None, [256], [0, 256])

        # Olasılık dağılımını (P_i) hesapla
        hist = hist.ravel() / hist.sum()

        # log2(0) hatasını önlemek için sadece sıfırdan büyük olasılıkları al
        nonzero_probs = hist[hist > 0]

        # Entropi formülünün (H = -sum(P * log2(P))) vektörize uygulanması
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))

        return variance, entropy

    def get_image_generator(self):
        """
        Bellek şişmesini önlemek için görüntüleri tek tek döndüren generator (yield).
        """
        for path in self.image_paths:
            img = self.load_image(path)
            var, ent = self.calculate_base_metrics(img)
            filename = os.path.basename(path)
            yield filename, img, var, ent
