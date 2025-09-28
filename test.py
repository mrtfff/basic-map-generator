import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter.ttk import Separator
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- 1. Çekirdek Harita Oluşturma Mantığı ---
# Bu kısım önceki kodumuzdan alındı ve bir fonksiyona dönüştürüldü.

def subdivide_and_displace(points, iterations, displacement_factor):
    if iterations == 0:
        return points
    new_points = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        segment_length = np.sqrt(dx**2 + dy**2)
        offset = (np.random.rand() - 0.5) * segment_length * displacement_factor
        normal_vector = np.array([-dy, dx])
        if np.linalg.norm(normal_vector) > 0:
            normal_vector /= np.linalg.norm(normal_vector)
        displaced_mid_x = mid_x + normal_vector[0] * offset
        displaced_mid_y = mid_y + normal_vector[1] * offset
        new_points.append(p1)
        new_points.append((displaced_mid_x, displaced_mid_y))
    new_points.append(points[-1])
    return subdivide_and_displace(new_points, iterations - 1, displacement_factor)

def create_island_shape(num_points, x_spread, y_spread, roughness):
    """Verilen parametrelerle bir ada şekli (nokta listesi) oluşturur."""
    # Gauss dağılımı ile noktalar oluştur
    points = np.zeros((num_points, 2))
    points[:, 0] = np.random.normal(loc=500, scale=x_spread, size=num_points)
    points[:, 1] = np.random.normal(loc=500, scale=y_spread, size=num_points)
    
    # Konveks Zarf
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_points_closed = np.vstack([hull_points, hull_points[0]])
    
    # Fraktal Pürüzlendirme
    fractal_iterations = int(roughness)
    displacement_factor = 0.3
    
    final_points = []
    for i in range(len(hull_points_closed) - 1):
        segment_points = subdivide_and_displace([hull_points_closed[i], hull_points_closed[i+1]], fractal_iterations, displacement_factor)
        final_points.extend(segment_points[:-1])
    final_points.append(final_points[0]) # Poligonu kapat
    
    return final_points


# --- 2. Tkinter Uygulama Sınıfı ---

class MapGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Prosedürel Ada Oluşturucu")
        self.root.geometry("1000x700")

        self.saves_file = "saves.json"
        self.saved_maps = {}
        self.current_points = None
        self.current_settings = {}

        # Ana Arayüz Panelleri
        self.control_frame = tk.Frame(root, width=250, bg="#f0f0f0")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.control_frame.pack_propagate(False) # Frame'in küçülmesini engelle

        self.map_frame = tk.Frame(root)
        self.map_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Matplotlib Figürünü Gömme
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,expand=True)
        # Pan/Zoom etkileşimleri
        self._init_interactions()
        
        # Arayüz elemanlarını oluştur
        self.create_controls()
        
        # Kayıtları yükle
        self.load_saves()
        
        # Başlangıçta boş bir harita göster
        self.draw_map(None)

    def create_controls(self):
        """Kontrol panelindeki tüm widget'ları oluşturur."""
        
        # --- Ayar Slider'ları ---
        tk.Label(self.control_frame, text="Harita Ayarları", font=("Helvetica", 14, "bold"), bg="#f0f0f0").pack(pady=10)

        tk.Label(self.control_frame, text="Yatay Yayılım:", bg="#f0f0f0").pack(pady=(10,0))
        self.x_spread = tk.Scale(self.control_frame, from_=50, to=3000, orient=tk.HORIZONTAL, bg="#f0f0f0", length=200)
        self.x_spread.set(600)
        self.x_spread.pack()

        tk.Label(self.control_frame, text="Dikey Yayılım:", bg="#f0f0f0").pack(pady=(10,0))
        self.y_spread = tk.Scale(self.control_frame, from_=50, to=3000, orient=tk.HORIZONTAL, bg="#f0f0f0", length=200)
        self.y_spread.set(600)
        self.y_spread.pack()

        tk.Label(self.control_frame, text="Nokta Sayısı:", bg="#f0f0f0").pack(pady=(10,0))
        self.num_points = tk.Scale(self.control_frame, from_=200, to=20000, orient=tk.HORIZONTAL, bg="#f0f0f0", length=200)
        self.num_points.set(5000)
        self.num_points.pack()

        tk.Label(self.control_frame, text="Pürüzlülük (Detay):", bg="#f0f0f0").pack(pady=(10,0))
        self.roughness = tk.Scale(self.control_frame, from_=1, to=8, orient=tk.HORIZONTAL, bg="#f0f0f0", length=200)
        self.roughness.set(5)
        self.roughness.pack()

        # --- Butonlar ---
        run_button = tk.Button(self.control_frame, text="Harita Oluştur", command=self.generate_map)
        run_button.pack(pady=10, fill=tk.X, padx=10)

        tk.Label(self.control_frame, text="Kayıt Adı:", bg="#f0f0f0").pack(pady=(5,0), padx=10, anchor='w')
        self.name_entry = tk.Entry(self.control_frame)
        self.name_entry.pack(fill=tk.X, padx=10)

        save_button = tk.Button(self.control_frame, text="Kaydet", command=self.save_current_map)
        save_button.pack(pady=(10, 10), fill=tk.X, padx=10)

        # Harita düzenleyici penceresi
        edit_button = tk.Button(self.control_frame, text="Haritayı Düzenle", command=self.open_editor)
        edit_button.pack(pady=(0, 10), fill=tk.X, padx=10)

        Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=10)

        # --- Kayıtlı Haritalar Listesi ---
        tk.Label(self.control_frame, text="Kayıtlı Haritalar", font=("Helvetica", 14, "bold"), bg="#f0f0f0").pack(pady=10)
        
        saves_frame = tk.Frame(self.control_frame)
        saves_frame.pack(fill=tk.BOTH, expand=True)
        self.saves_listbox = tk.Listbox(saves_frame)
        self.saves_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(saves_frame, orient=tk.VERTICAL)
        scrollbar.config(command=self.saves_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.saves_listbox.config(yscrollcommand=scrollbar.set)
        self.saves_listbox.bind('<<ListboxSelect>>', self.on_save_selected)

    def generate_map(self):
        """Slider'lardan ayarları al ve haritayı oluştur/çiz (kaydetmez)."""
        num = self.num_points.get()
        x_s = self.x_spread.get()
        y_s = self.y_spread.get()
        rough = self.roughness.get()

        # Haritayı oluştur
        final_points = create_island_shape(num, x_s, y_s, rough)
        
        # Haritayı çiz ve güncel durumu sakla
        self.draw_map(final_points)
        self.current_points = np.array(final_points)
        self.current_settings = {
            'num_points': num,
            'x_spread': x_s,
            'y_spread': y_s,
            'roughness': rough
        }

    def save_current_map(self):
        """Mevcut haritayı (nokta listesi + ayarlar) isimle kaydeder."""
        if self.current_points is None:
            messagebox.showwarning("Uyarı", "Önce bir harita oluşturun.")
            return

        map_name = self.name_entry.get().strip()
        if not map_name:
            messagebox.showwarning("Geçersiz İsim", "Lütfen bir kayıt adı girin.")
            return

        # Kaydetme yapısı: { name: { 'points': ndarray, 'settings': dict } }
        self.saved_maps[map_name] = {
            'points': np.array(self.current_points),
            'settings': dict(self.current_settings)
        }
        self.update_saves_listbox()
        self.save_to_file()
        messagebox.showinfo("Başarılı", f"'{map_name}' adıyla harita kaydedildi.")

    def generate_and_save_map(self):
        """GERİYE DÖNÜK UYUMLULUK: Eski butonun çağırabileceği fonksiyon. Artık yalnızca harita oluşturur."""
        # Eski davranıştaki otomatik kaydetme kaldırıldı.
        self.generate_map()

    def open_editor(self):
        """Seçili veya mevcut haritayı düzenlemek için düzenleyici penceresini açar."""
        # Kaynak noktaları belirle
        points = None
        settings = {}
        selected_name = None
        sel = self.saves_listbox.curselection()
        if sel:
            selected_name = self.saves_listbox.get(sel[0])
            item = self.saved_maps[selected_name]
            if isinstance(item, dict) and 'points' in item:
                points = np.array(item['points'])
                settings = dict(item.get('settings', {}) or {})
            else:
                points = np.array(item)
                settings = {}
        elif self.current_points is not None:
            points = np.array(self.current_points)
            settings = dict(self.current_settings or {})
        else:
            messagebox.showwarning("Uyarı", "Önce bir harita oluşturun veya listeden bir kayıt seçin.")
            return

        if points is None or len(points) < 3:
            messagebox.showwarning("Uyarı", "Düzenlemek için geçerli bir harita bulunamadı.")
            return

        MapEditorWindow(self, points, settings, selected_name)

    def draw_map(self, points):
        """Verilen nokta listesine göre haritayı çizer."""
        self.ax.clear()
        self.ax.set_facecolor('#6CA0DC') # Su rengi
        if points is not None and len(points) > 0:
            landmass = Polygon(points, closed=True, facecolor='#556B2F', edgecolor='black', linewidth=1)
            self.ax.add_patch(landmass)
            try:
                pts = np.array(points)
                min_x, max_x = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
                min_y, max_y = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
                # Kenarlarda pay bırak
                x_range = max(1.0, max_x - min_x)
                y_range = max(1.0, max_y - min_y)
                pad_x = x_range * 0.05
                pad_y = y_range * 0.05
                self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
                self.ax.set_ylim(min_y - pad_y, max_y + pad_y)
            except Exception:
                # Her ihtimale karşın varsayılan limitler
                self.ax.set_xlim(0, 1000)
                self.ax.set_ylim(0, 1000)
        else:
            # İlk açılış veya boş durumda varsayılan alan
            self.ax.set_xlim(0, 1000)
            self.ax.set_ylim(0, 1000)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('off')
        self.canvas.draw()

    # --- Pan/Zoom Etkileşimleri ---
    def _init_interactions(self):
        self._is_panning = False
        self._press_event = None
        self._press_xlim = None
        self._press_ylim = None
        self._last_xy = None
        # Matplotlib olaylarını bağla
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        # Sol tık ile pan başlat
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self._is_panning = True
            self._press_event = (event.xdata, event.ydata)
            self._press_xlim = self.ax.get_xlim()
            self._press_ylim = self.ax.get_ylim()
            self._last_xy = (event.xdata, event.ydata)

    def _on_release(self, event):
        # Pan'i bırak
        self._is_panning = False
        self._press_event = None
        self._press_xlim = None
        self._press_ylim = None
        self._last_xy = None

    def _on_motion(self, event):
        if not self._is_panning or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None or self._last_xy is None:
            return
        x_prev, y_prev = self._last_xy
        dx = event.xdata - x_prev
        dy = event.ydata - y_prev
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        new_xlim = (cur_xlim[0] - dx, cur_xlim[1] - dx)
        new_ylim = (cur_ylim[0] - dy, cur_ylim[1] - dy)
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self._last_xy = (event.xdata, event.ydata)
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        # Fare konumuna göre yakınlaştır/uzaklaştır (daha yumuşak oran)
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2
        ydata = event.ydata if event.ydata is not None else (cur_ylim[0] + cur_ylim[1]) / 2

        # Zoom faktörü (daha küçük adım -> daha akıcı zoom)
        zoom_in = None
        if hasattr(event, 'step'):
            zoom_in = event.step > 0
        else:
            if getattr(event, 'button', None) == 'up':
                zoom_in = True
            elif getattr(event, 'button', None) == 'down':
                zoom_in = False
        if zoom_in is None:
            zoom_in = True

        scale_factor = 0.95 if zoom_in else 1.05

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        # Minimum görünüm sınırı (çok fazla zoom ile sarsılmayı engelle)
        min_span = 0.2
        new_width = max(new_width, min_span)
        new_height = max(new_height, min_span)

        width = max(cur_xlim[1] - cur_xlim[0], 1e-9)
        height = max(cur_ylim[1] - cur_ylim[0], 1e-9)
        relx = min(max((xdata - cur_xlim[0]) / width, 0.0), 1.0)
        rely = min(max((ydata - cur_ylim[0]) / height, 0.0), 1.0)

        target_xlim = (xdata - new_width * relx, xdata + new_width * (1 - relx))
        target_ylim = (ydata - new_height * rely, ydata + new_height * (1 - rely))

        # Sarsılmayı azaltmak için eksen limitlerini yumuşatarak uygula
        alpha = 0.35
        smoothed_xlim = (
            cur_xlim[0] * (1 - alpha) + target_xlim[0] * alpha,
            cur_xlim[1] * (1 - alpha) + target_xlim[1] * alpha,
        )
        smoothed_ylim = (
            cur_ylim[0] * (1 - alpha) + target_ylim[0] * alpha,
            cur_ylim[1] * (1 - alpha) + target_ylim[1] * alpha,
        )

        self.ax.set_xlim(smoothed_xlim)
        self.ax.set_ylim(smoothed_ylim)
        self.canvas.draw_idle()
        self.canvas.draw()
        
    def on_save_selected(self, event):
        """Listeden bir kayıt seçildiğinde onu ekranda gösterir ve ayarları uygular."""
        selected_indices = self.saves_listbox.curselection()
        if not selected_indices:
            return
        
        selected_name = self.saves_listbox.get(selected_indices[0])
        item = self.saved_maps[selected_name]

        # Geriye dönük uyumluluk: Eski format list/ndarray ise
        if isinstance(item, dict) and 'points' in item:
            points_to_draw = item['points']
            settings = item.get('settings', {}) or {}
        else:
            points_to_draw = item
            settings = {}

        self.draw_map(points_to_draw)
        self.current_points = np.array(points_to_draw)
        self.current_settings = dict(settings)

        # Ayarları slider'lara uygula (varsa)
        if settings:
            if 'x_spread' in settings:
                self.x_spread.set(settings['x_spread'])
            if 'y_spread' in settings:
                self.y_spread.set(settings['y_spread'])
            if 'num_points' in settings:
                self.num_points.set(settings['num_points'])
            if 'roughness' in settings:
                self.roughness.set(settings['roughness'])
        
        # Seçilen adı girişe yaz
        if hasattr(self, 'name_entry'):
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, selected_name)

    def update_saves_listbox(self):
        """Kayıt listesini günceller."""
        self.saves_listbox.delete(0, tk.END)
        for name in sorted(self.saved_maps.keys()):
            self.saves_listbox.insert(tk.END, name)

    def save_to_file(self):
        """Tüm kayıtları JSON dosyasına yazar (ayarlarla birlikte)."""
        with open(self.saves_file, 'w') as f:
            data_to_save = {}
            for name, item in self.saved_maps.items():
                if isinstance(item, dict) and 'points' in item:
                    data_to_save[name] = {
                        'points': np.array(item['points']).tolist(),
                        'settings': item.get('settings', {})
                    }
                else:
                    # Eski formatı olduğu gibi yaz (list)
                    data_to_save[name] = np.array(item).tolist()
            json.dump(data_to_save, f, indent=4)
            
    def load_saves(self):
        """JSON dosyasından kayıtları yükler (eski/yeni format uyumlu)."""
        try:
            with open(self.saves_file, 'r') as f:
                loaded_data = json.load(f)
                self.saved_maps = {}
                for name, val in loaded_data.items():
                    if isinstance(val, dict) and 'points' in val:
                        self.saved_maps[name] = {
                            'points': np.array(val['points']),
                            'settings': val.get('settings', {}) or {}
                        }
                    else:
                        # Eski format: sadece nokta listesi
                        self.saved_maps[name] = np.array(val)
            self.update_saves_listbox()
        except FileNotFoundError:
            # Dosya yoksa sorun değil, ilk kayıtta oluşturulacak.
            pass
        except json.JSONDecodeError:
            messagebox.showerror("Hata", f"{self.saves_file} dosyası bozuk veya boş.")

# --- 3. Uygulamayı Başlatma ---

if __name__ == "__main__":
    root = tk.Tk()
    app = MapGeneratorApp(root)
    root.mainloop()

class MapEditorWindow:
    def __init__(self, app: MapGeneratorApp, points: np.ndarray, settings: dict, save_name: str | None):
        self.app = app
        self.root = tk.Toplevel(app.root)
        self.root.title("Harita Düzenleyici - Çıkıntı/Girinti")
        self.root.geometry("900x700")
        self.save_name = save_name
        self.settings = dict(settings or {})

        # Çalışma kopyası (kapalı poligon olarak tut)
        self.points = self._ensure_closed(np.array(points, dtype=float))
        self.undo_stack = []

        # Yerleşim
        self.left = tk.Frame(self.root, width=220, bg="#f7f7f7")
        self.left.pack(side=tk.LEFT, fill=tk.Y)
        self.left.pack_propagate(False)
        self.right = tk.Frame(self.root)
        self.right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Kontroller
        tk.Label(self.left, text="Düzenleme Modu", bg="#f7f7f7", font=("Helvetica", 12, "bold")).pack(pady=(10, 5))
        self.mode_var = tk.StringVar(value="protrusion")
        tk.Radiobutton(self.left, text="Çıkıntı", variable=self.mode_var, value="protrusion", bg="#f7f7f7").pack(anchor='w', padx=10)
        tk.Radiobutton(self.left, text="Girinti", variable=self.mode_var, value="indentation", bg="#f7f7f7").pack(anchor='w', padx=10)

        tk.Label(self.left, text="Şiddet (%)", bg="#f7f7f7").pack(pady=(10,0), anchor='w', padx=10)
        self.intensity = tk.Scale(self.left, from_=5, to=60, orient=tk.HORIZONTAL, length=180, bg="#f7f7f7")
        self.intensity.set(25)
        self.intensity.pack(padx=10, pady=(0,10))

        tk.Button(self.left, text="Geri Al", command=self.undo).pack(fill=tk.X, padx=10, pady=(5,5))
        tk.Button(self.left, text="Kaydet", command=self.save).pack(fill=tk.X, padx=10, pady=(5,5))
        tk.Button(self.left, text="Farklı Kaydet...", command=self.save_as).pack(fill=tk.X, padx=10, pady=(0,10))
        tk.Button(self.left, text="Kapat", command=self.root.destroy).pack(fill=tk.X, padx=10, pady=(0,10))

        # Matplotlib tuvali
        self.fig = Figure(figsize=(6,6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        self._draw()

    # --- Yardımcılar ---
    def _ensure_closed(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        return pts

    def _draw(self):
        self.ax.clear()
        self.ax.set_facecolor('#6CA0DC')
        if self.points is not None and len(self.points) > 2:
            poly = Polygon(self.points, closed=True, facecolor='#556B2F', edgecolor='black', linewidth=1)
            self.ax.add_patch(poly)
            xs = self.points[:,0]
            ys = self.points[:,1]
            xr = max(1.0, float(xs.max() - xs.min()))
            yr = max(1.0, float(ys.max() - ys.min()))
            pad_x = xr * 0.06
            pad_y = yr * 0.06
            self.ax.set_xlim(float(xs.min()) - pad_x, float(xs.max()) + pad_x)
            self.ax.set_ylim(float(ys.min()) - pad_y, float(ys.max()) + pad_y)
        else:
            self.ax.set_xlim(0, 1000)
            self.ax.set_ylim(0, 1000)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('off')
        self.canvas.draw()

    def _nearest_edge_index(self, x: float, y: float) -> int:
        p = np.array([x, y])
        pts = self.points
        n = len(pts)
        best_i = 0
        best_d = float('inf')
        for i in range(n-1):
            a = pts[i]
            b = pts[i+1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 == 0:
                d = float(np.linalg.norm(p - a))
            else:
                t = float(np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0))
                proj = a + t * ab
                d = float(np.linalg.norm(p - proj))
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        # Geri alma için yedekle
        self.undo_stack.append(self.points.copy())

        i = self._nearest_edge_index(event.xdata, event.ydata)
        pts = self.points.copy()
        p1 = pts[i]
        p2 = pts[i+1]
        e = p2 - p1
        elen = float(np.linalg.norm(e))
        if elen == 0:
            return
        m = (p1 + p2) / 2.0
        normal = np.array([-e[1], e[0]]) / elen
        centroid = np.mean(pts[:-1], axis=0)
        delta = (self.intensity.get() / 100.0) * elen
        cand1 = m + normal * delta
        cand2 = m - normal * delta
        # Mod yönü seçimi
        if self.mode_var.get() == 'protrusion':
            # Dışarı: merkezden uzak olan
            d1 = np.linalg.norm(cand1 - centroid)
            d2 = np.linalg.norm(cand2 - centroid)
            apex = cand1 if d1 >= d2 else cand2
        else:
            # İçeri: merkeze yakın olan
            d1 = np.linalg.norm(cand1 - centroid)
            d2 = np.linalg.norm(cand2 - centroid)
            apex = cand1 if d1 <= d2 else cand2
        # Taban genişliği için iki ara nokta
        t1, t2 = 0.35, 0.65
        q1 = p1 * (1 - t1) + p2 * t1
        q2 = p1 * (1 - t2) + p2 * t2
        # İnsersiyon: i+1 konumuna [q1, apex, q2]
        new_pts = np.vstack([pts[:i+1], q1, apex, q2, pts[i+1:]])
        new_pts = self._ensure_closed(new_pts[:-1])  # yeniden kapat
        self.points = new_pts
        self._draw()

    def undo(self):
        if not self.undo_stack:
            return
        self.points = self.undo_stack.pop()
        self._draw()

    def save(self):
        name = self.save_name
        if not name:
            # Eğer mevcut bir kayda bağlı değilse, farklı kaydet iste
            return self.save_as()
        self._commit_save(name)

    def save_as(self):
        default = self.save_name or "duzenlenmis_harita"
        name = simpledialog.askstring("Farklı Kaydet", "Yeni kayıt adı:", initialvalue=default, parent=self.root)
        if not name:
            return
        self._commit_save(name)

    def _commit_save(self, name: str):
        # Ayarları koru (varsa), yoksa app.current_settings kullan
        settings = dict(self.settings or self.app.current_settings or {})
        self.app.saved_maps[name] = {
            'points': np.array(self.points).tolist(),
            'settings': settings
        }
        self.app.update_saves_listbox()
        self.app.save_to_file()
        self.app.name_entry.delete(0, tk.END)
        self.app.name_entry.insert(0, name)
        # Ana görünümü güncelle
        self.app.draw_map(self.points)
        self.app.current_points = np.array(self.points)
        self.app.current_settings = settings
        messagebox.showinfo("Kaydedildi", f"'{name}' güncellendi.")