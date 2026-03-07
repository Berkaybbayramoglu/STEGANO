import numpy as np
from skimage.metrics import structural_similarity as ssim


class StegoEvaluator:
    def __init__(self):
        """
        Steganografi performans metriklerini hesaplayan ve veri gömen değerlendirme modülü.
        """
        pass

    def embed_lsb(self, original_image, pixel_coords, payload_bits=None):
        """
        Verilen koordinatlardaki piksellerin En Az Anlamlı Bitine (LSB) veriyi gömer.
        """
        # Orijinal matrisi bozmamak için kopyasını alıyoruz (Stego-Görüntü olacak)
        stego_image = original_image.copy()
        L = len(pixel_coords)

        # Eğer dışarıdan bir veri (mesaj) verilmediyse, test için rastgele 0 ve 1'ler üret
        if payload_bits is None:
            payload_bits = np.random.randint(0, 2, size=L, dtype=np.uint8)

        # Piksellere bitleri yerleştir
        for idx, (y, x) in enumerate(pixel_coords):
            pixel_val = stego_image[y, x]
            bit = payload_bits[idx]

            # Bitwise işlemi:
            # & 254 (11111110) ile sayının son bitini 0 yaparız (Temizleme)
            # | bit (0 veya 1) ile mesaj bitimizi yerleştiririz
            stego_image[y, x] = (pixel_val & 254) | bit

        return stego_image

    def calculate_metrics(self, original_image, stego_image):
        """
        PSNR ve SSIM metriklerini hesaplar. Makale tabloları için kritik kısımdır.
        """
        # Hata Kareler Ortalaması (MSE)
        mse = np.mean(
            (original_image.astype(np.float64) - stego_image.astype(np.float64)) ** 2
        )

        # PSNR Hesaplaması
        if mse == 0:
            psnr = float("inf")  # Görüntüler birebir aynıysa
        else:
            max_pixel = 255.0
            psnr = 10 * np.log10((max_pixel**2) / mse)

        # SSIM Hesaplaması (Piksel aralığı 8-bit görüntüler için 255'tir)
        ssim_val = ssim(original_image, stego_image, data_range=255)

        return mse, psnr, ssim_val
