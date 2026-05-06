import numpy as np


class ABCOptimizer:
    def __init__(
        self, safe_pixels, payload_size, colony_size=20, max_iter=50, limit=15
    ):
        """
        Steganografi için özelleştirilmiş Ayrık (Discrete) Yapay Arı Kolonisi Algoritması.

        :param safe_pixels: Quadtree'den gelen [(y1, x1), (y2, x2)...] formatındaki güvenli pikseller.
        :param payload_size: Gizlenecek verinin boyutu (L). Seçilecek piksel sayısı.
        :param colony_size: Toplam arı sayısı. (Yarısı işçi, yarısı gözcü)
        :param max_iter: Maksimum jenerasyon (döngü) sayısı.
        :param limit: Bir yiyecek kaynağı 'limit' kadar gelişemezse kaşif arı tarafından terk edilir.
        """
        self.safe_pixels = np.array(safe_pixels)
        self.K = len(self.safe_pixels)  # Toplam güvenli piksel havuzu
        self.L = payload_size  # Çözüm vektörünün uzunluğu

        self.num_employed = colony_size // 2
        self.max_iter = max_iter
        self.limit = limit

        # Popülasyon Matrisleri
        # Her satır bir yiyecek kaynağıdır (L adet indeks içerir)
        self.foods = np.zeros((self.num_employed, self.L), dtype=int)
        self.fitness = np.zeros(self.num_employed)
        self.trial_counters = np.zeros(self.num_employed)

        self.best_food = None
        self.best_fitness = -1.0

    def _calculate_fitness(self, food_indices, image_matrix):
        """
        GEÇİCİ UYGUNLUK FONKSİYONU:
        Seçilen piksellerin orijinal görüntüdeki varyans/kenar yoğunluğunu değerlendirir.
        Gömme işlemi eklendiğinde buraya MSE/PSNR cezası da eklenecek.
        """
        selected_coords = self.safe_pixels[food_indices]

        score = 0.0
        for y, x in selected_coords:
            score += image_matrix[y, x]

        # Fitness maksimize edilmek istendiği için doğrudan skoru dönüyoruz
        return score

    def _get_neighbor(self, current_food):
        """
        Ayrık uzayda yeni bir komşu çözüm (neighbor) üretir.
        Mevcut çözümden rastgele 1 pikseli atıp yerine havuzdan yeni bir piksel koyar.
        """
        neighbor = current_food.copy()

        dim_to_change = np.random.randint(0, self.L)  # Hangi pikseli değiştireceğiz

        while True:
            new_pixel_idx = np.random.randint(self.L)
            if new_pixel_idx not in neighbor:
                neighbor[dim_to_change] = new_pixel_idx
                break
        return neighbor

    def initialize_population(self, image_matrix):
        """
        Başlangıç popülasyonunu rastgele oluşturur.
        """
        for i in range(self.num_employed):
            # K adet güvenli pikselden, L adet benzersiz indeks seç (Yerine koymadan - replace=False)
            self.foods[i] = np.random.choice(self.K, self.L, replace=False)
            self.fitness[i] = self._calculate_fitness(self.foods[i], image_matrix)

            # En iyi çözümü güncelle
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_food = self.foods[i].copy()

        print(
            f"Popülasyon başlatıldı. Başlangıç En İyi Uygunluk: {self.best_fitness:.2f}"
        )

    def optimize(self, image_matrix):
        """
        İşçi, Gözcü ve Kaşif arı fazlarını çalıştıracak ana optimizasyon döngüsü.
        """
        self.initialize_population(image_matrix)

        for it in range(self.max_iter):
            # ---------------------------------------------------------
            # 1. İŞÇİ ARI FAZI (Employed Bees)
            # ---------------------------------------------------------
            for i in range(self.num_employed):
                # Komşu üret ve uygunluğunu ölç
                neighbor = self._get_neighbor(self.foods[i])
                neighbor_fitness = self._calculate_fitness(neighbor, image_matrix)

                # Açgözlü Seçim (Greedy Selection)
                if neighbor_fitness > self.fitness[i]:
                    self.foods[i] = neighbor
                    self.fitness[i] = neighbor_fitness
                    self.trial_counters[i] = 0  # Gelişim sağlandı, sayacı sıfırla
                else:
                    self.trial_counters[i] += 1

            # ---------------------------------------------------------
            # 2. GÖZCÜ ARI FAZI (Onlooker Bees)
            # ---------------------------------------------------------
            # Rulet tekerleği için seçim olasılıklarını hesapla
            total_fitness = np.sum(self.fitness)
            if total_fitness == 0:
                probabilities = np.ones(self.num_employed) / self.num_employed
            else:
                probabilities = self.fitness / total_fitness

            t = 0
            i = 0
            # Gözcü arı sayısı kadar (İşçi sayısına eşittir) seçim yap
            while t < self.num_employed:
                # Rulet tekerleği: Olasılığı yüksek olan yiyecek daha çok seçilir
                if np.random.rand() < probabilities[i]:
                    t += 1
                    neighbor = self._get_neighbor(self.foods[i])
                    neighbor_fitness = self._calculate_fitness(neighbor, image_matrix)

                    if neighbor_fitness > self.fitness[i]:
                        self.foods[i] = neighbor
                        self.fitness[i] = neighbor_fitness
                        self.trial_counters[i] = 0
                    else:
                        self.trial_counters[i] += 1

                i = (i + 1) % self.num_employed

            # Her iki fazdan sonra en iyi genel çözümü (Global Best) güncelle
            best_idx = np.argmax(self.fitness)
            if self.fitness[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_food = self.foods[best_idx].copy()

            # ---------------------------------------------------------
            # 3. KAŞİF ARI FAZI (Scout Bees)
            # ---------------------------------------------------------
            for i in range(self.num_employed):
                # Eğer bir yiyecek kaynağı "limit" boyunca gelişemediyse terk et
                if self.trial_counters[i] >= self.limit:
                    self.foods[i] = np.random.choice(self.K, self.L, replace=False)
                    self.fitness[i] = self._calculate_fitness(
                        self.foods[i], image_matrix
                    )
                    self.trial_counters[i] = 0  # Yeni kaynak, sayacı sıfırla

            # Süreci takip etmek için log basımı
            if (it + 1) % 10 == 0:
                print(
                    f"  -> İterasyon {it + 1:03d}/{self.max_iter} | Güncel En İyi Uygunluk: {self.best_fitness:.2f}"
                )

        # Optimizasyon bittiğinde, en iyi yiyecek kaynağının tuttuğu (y,x) koordinatlarını döndür
        return self.safe_pixels[self.best_food]
