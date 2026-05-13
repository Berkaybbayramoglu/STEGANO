import cv2
import numpy as np

def draw_quadtree_art(img_color, img_gray, x, y, w, h, canvas, variance_threshold, min_size):
    # Karmaşıklığı (varyansı) bulmak için gri tonlamalı bölgeyi kullanıyoruz
    block_gray = img_gray[y:y+h, x:x+w]
    if block_gray.size == 0:
        return

    variance = np.var(block_gray)

    # Eğer blok karmaşıksa ve minimum boyuttan büyükse dörde böl
    if variance > variance_threshold and w > min_size and h > min_size:
        half_w, half_h = w // 2, h // 2
        
        draw_quadtree_art(img_color, img_gray, x, y, half_w, half_h, canvas, variance_threshold, min_size)
        draw_quadtree_art(img_color, img_gray, x + half_w, y, half_w, half_h, canvas, variance_threshold, min_size)
        draw_quadtree_art(img_color, img_gray, x, y + half_h, half_w, half_h, canvas, variance_threshold, min_size)
        draw_quadtree_art(img_color, img_gray, x + half_w, y + half_h, half_w, half_h, canvas, variance_threshold, min_size)
    else:
        # Renkleri orijinal resimden (renkli kısımdan) alıyoruz
        block_color = img_color[y:y+h, x:x+w]
        
        # Bloğun ortalama BGR (Mavi, Yeşil, Kırmızı) renk değerini hesapla
        mean_color = tuple(int(c) for c in cv2.mean(block_color)[:3])
        
        # Kutucuğun içini o renkle doldur
        cv2.rectangle(canvas, (x, y), (x+w, y+h), mean_color, -1)
        
        # Kutucuk efekti için ince siyah çerçeve
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 0, 0), 1)

def main():
    image_path = "STEGANO.jpg" 
    
    # 1. Resmi bu sefer RENKLİ okuyoruz
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if img_color is None:
        print("HATA: STEGANO.jpg dosyası bulunamadı!")
        return

    # Quadtree analizini yapabilmek için arka planda grisine ihtiyacımız var
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Arka planı beyaz olacak boş bir tuval oluştur
    canvas = np.ones_like(img_color) * 255
    
    print("Renkli Quadtree sanat eseri üretiliyor...")
    
    # (Eğer kutucuklar çok büyük veya küçük gelirse 30 ve 4 sayılarıyla oynayabilirsin)
    draw_quadtree_art(img_color, img_gray, 0, 0, img_color.shape[1], img_color.shape[0], canvas, variance_threshold=30, min_size=4)
    
    # 3. Sonucu kaydet
    output_path = "quadtree_ari_renkli.jpg"
    cv2.imwrite(output_path, canvas)
    print(f"Bitti! YENİ RENKLİ resminiz kaydedildi: {output_path}")

if __name__ == "__main__":
    main()