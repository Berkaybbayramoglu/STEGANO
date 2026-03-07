import numpy as np


class QuadTreeDecomposer:
    def __init__(self, min_block_size=4, variance_threshold=50.0):
        """
        Görüntüyü Quadtree mantığıyla bölen sınıf.
        :param min_block_size: İneceğimiz en küçük blok boyutu (örn: 4x4)
        :param variance_threshold: Bir bloğun 'karmaşık' sayılması için gereken minimum varyans (T)
        """
        self.min_block_size = min_block_size
        self.variance_threshold = variance_threshold

    def decompose(self, image_matrix):
        """
        Görüntüyü analiz eder ve sadece güvenli (karmaşık) piksellerin koordinatlarını döndürür.
        """
        # Güvenli piksellerin koordinatlarını tutacağımız liste (Sparse Matrix mantığı)
        valid_pixels = []
        h, w = image_matrix.shape

        # Özyinelemeli bölme işlemini başlat
        self._divide(image_matrix, 0, 0, w, h, valid_pixels)

        # Listeyi numpy dizisine çevirerek performansı artırıyoruz
        return np.array(valid_pixels)

    def _divide(self, img, x, y, w, h, valid_pixels):
        # Geçerli bloğu al
        block = img[y : y + h, x : x + w]

        # Bloğun varyansını hesapla
        variance = np.var(block)

        # Durum 1: Blok düz bir alan (Varyans düşük). Hiçbir şey yapma, bloğu çöpe at.
        if variance < self.variance_threshold:
            return

        # Durum 2: Blok karmaşık (Varyans yüksek)
        if w > self.min_block_size and h > self.min_block_size:
            # Bloğu dörde böl (Quadtree yapısı)
            half_w, half_h = w // 2, h // 2

            self._divide(img, x, y, half_w, half_h, valid_pixels)  # Sol-Üst
            self._divide(img, x + half_w, y, half_w, half_h, valid_pixels)  # Sağ-Üst
            self._divide(img, x, y + half_h, half_w, half_h, valid_pixels)  # Sol-Alt
            self._divide(
                img, x + half_w, y + half_h, half_w, half_h, valid_pixels
            )  # Sağ-Alt
        else:
            # En küçük alt bloğa ulaştık ve hala karmaşık. Pikselleri güvenli olarak kaydet.
            for i in range(y, y + h):
                for j in range(x, x + w):
                    valid_pixels.append((i, j))
