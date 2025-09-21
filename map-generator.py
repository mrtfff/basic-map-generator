import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# --- Aşama 3 için yardımcı fonksiyon ---
# Bu fonksiyon, bir çizgi parçasını alır ve onu pürüzlü hale getirir.
def subdivide_and_displace(points, iterations, displacement_factor=0.3):
    """
    Bir çizgi segment listesini fraktal olarak böler ve kaydırır.
    points: Başlangıç ve bitiş noktaları [(x1, y1), (x2, y2), ...]
    iterations: Kaç kez bölüneceği (recursion derinliği)
    displacement_factor: Kaydırma miktarının segment uzunluğuna oranı
    """
    if iterations == 0:
        return points

    new_points = []
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        
        # İki nokta arasına yeni bir orta nokta ekle
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        
        # Kaydırma miktarını hesapla
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        segment_length = np.sqrt(dx**2 + dy**2)
        offset = (np.random.rand() - 0.5) * segment_length * displacement_factor
        
        # Orta noktayı segmente dik olarak kaydır
        # Dik vektör (-dy, dx)
        normal_vector = np.array([-dy, dx])
        normal_vector /= np.linalg.norm(normal_vector) # Birim vektör yap
        
        displaced_mid_x = mid_x + normal_vector[0] * offset
        displaced_mid_y = mid_y + normal_vector[1] * offset
        
        new_points.append(p1)
        new_points.append((displaced_mid_x, displaced_mid_y))
    
    new_points.append(points[-1])
    
    # Bir sonraki iterasyon için tekrar çağır
    return subdivide_and_displace(new_points, iterations - 1, displacement_factor)


# --- ANA ALGORİTMA ---

# Aşama 1: Noktaların Oluşturulması
num_points = 1000
# Gauss dağılımı ile merkezde yoğunlaşan noktalar oluştur
# loc=merkez, scale=standart sapma (yayılma), size=(nokta sayısı, boyut)
points = np.random.normal(loc=500, scale=150, size=(num_points, 2))

# Aşama 2: Konveks Zarf'ın Bulunması
hull = ConvexHull(points)
hull_points = points[hull.vertices]
# Poligonu kapatmak için ilk noktayı sona ekle
hull_points_closed = np.vstack([hull_points, hull_points[0]])


# Aşama 3: Çizgilerin Doğallaştırılması
fractal_iterations = 4 # Detay seviyesi, artırarak daha pürüzlü yapabilirsiniz
final_points = []
# Zarfın her bir kenarı için pürüzlendirme işlemini yap
for i in range(len(hull_points_closed) - 1):
    p1 = hull_points_closed[i]
    p2 = hull_points_closed[i+1]
    
    # İlk noktayı atlayarak ekle, çünkü bir önceki segmentin son noktası zaten aynı
    segment_points = subdivide_and_displace([p1, p2], fractal_iterations)
    final_points.extend(segment_points[:-1])

final_points.append(final_points[0]) # Poligonu kapat


# Aşama 4: Görselleştirme
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor('#6CA0DC') # Su rengi (mavi)

# Kara parçasını çiz ve içini doldur
landmass = Polygon(final_points, closed=True, facecolor='#556B2F', edgecolor='black', linewidth=1)
ax.add_patch(landmass)

# İsteğe bağlı: Orijinal noktaları ve Konveks Zarf'ı göstermek için
# ax.plot(points[:, 0], points[:, 1], 'o', markersize=2, color='gray', alpha=0.5, label='Rastgele Noktalar')
# ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], 'r--', lw=2, label='Konveks Zarf')
# plt.legend()

ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_aspect('equal', adjustable='box')
plt.title("Prosedürel Olarak Oluşturulmuş Ada")
plt.axis('off') # Eksenleri gizle
plt.show()