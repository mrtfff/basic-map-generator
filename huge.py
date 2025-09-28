import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Separator
import json
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon

class HugeMapViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Devasa Harita Oluşturucu ve Görüntüleyici")
        self.root.geometry("1200x800")

        self.saves_file = "huge_saves.json"
        self.saved_maps = {}
        self.current_points = None
        self.current_settings = {}

        # Sol panel - ayarlar
        self.control_frame = tk.Frame(root, width=300, bg="#f7f7f7")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self.control_frame.pack_propagate(False)

        # Sağ panel - çizim
        self.map_frame = tk.Frame(root)
        self.map_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Matplotlib
        self.fig = Figure(figsize=(9, 9), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Çizim patch cache'i (yeniden oluşturmak yerine güncelleme için)
        self.poly_patch = None

        # Etkileşimler
        self._init_interactions()

        # Kontroller
        self._build_controls()

        # Kayıtlar
        self.load_saves()

        # Boş çizim
        self._draw(None)

    def _build_controls(self):
        tk.Label(self.control_frame, text="Dev Harita Ayarları", font=("Helvetica", 14, "bold"), bg="#f7f7f7").pack(pady=10)

        # Çok daha büyük parametre aralıkları
        tk.Label(self.control_frame, text="Nokta Sayısı (dev)", bg="#f7f7f7").pack(anchor='w', padx=10)
        self.num_points = tk.Scale(self.control_frame, from_=5000, to=200000, orient=tk.HORIZONTAL, length=240, bg="#f7f7f7")
        self.num_points.set(20000)
        self.num_points.pack(pady=(0,10))

        tk.Label(self.control_frame, text="Yatay Yayılım", bg="#f7f7f7").pack(anchor='w', padx=10)
        self.x_spread = tk.Scale(self.control_frame, from_=200, to=5000, orient=tk.HORIZONTAL, length=240, bg="#f7f7f7")
        self.x_spread.set(1500)
        self.x_spread.pack(pady=(0,10))

        tk.Label(self.control_frame, text="Dikey Yayılım", bg="#f7f7f7").pack(anchor='w', padx=10)
        self.y_spread = tk.Scale(self.control_frame, from_=200, to=5000, orient=tk.HORIZONTAL, length=240, bg="#f7f7f7")
        self.y_spread.set(1500)
        self.y_spread.pack(pady=(0,10))

        tk.Label(self.control_frame, text="Pürüzlülük (detay)", bg="#f7f7f7").pack(anchor='w', padx=10)
        self.roughness = tk.Scale(self.control_frame, from_=1, to=7, orient=tk.HORIZONTAL, length=240, bg="#f7f7f7")
        self.roughness.set(5)
        self.roughness.pack(pady=(0,10))

        Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=10)

        tk.Label(self.control_frame, text="Kayıt Adı:", bg="#f7f7f7").pack(anchor='w', padx=10)
        self.name_entry = tk.Entry(self.control_frame)
        self.name_entry.pack(fill=tk.X, padx=10, pady=(0,10))

        tk.Button(self.control_frame, text="Harita Oluştur (Dev)", command=self.generate).pack(fill=tk.X, padx=10, pady=5)
        tk.Button(self.control_frame, text="Kaydet", command=self.save_current).pack(fill=tk.X, padx=10, pady=5)

        Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=10)

        tk.Label(self.control_frame, text="Kayıtlı Dev Haritalar", font=("Helvetica", 12, "bold"), bg="#f7f7f7").pack(pady=(0,5))
        frame = tk.Frame(self.control_frame, bg="#f7f7f7")
        frame.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(frame)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(frame, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.bind('<<ListboxSelect>>', self.on_select)

    # --- Etkileşimler (pan/zoom) ---
    def _init_interactions(self):
        self._is_panning = False
        self._press_event = None
        self._press_xlim = None
        self._press_ylim = None
        self._last_xy = None
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self._is_panning = True
            self._press_event = (event.xdata, event.ydata)
            self._press_xlim = self.ax.get_xlim()
            self._press_ylim = self.ax.get_ylim()
            self._last_xy = (event.xdata, event.ydata)

    def _on_release(self, event):
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
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2
        ydata = event.ydata if event.ydata is not None else (cur_ylim[0] + cur_ylim[1]) / 2

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

        # Daha küçük adım -> daha akıcı zoom
        scale_factor = 0.95 if zoom_in else 1.05
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        # Aşırı yakınlaşmada sarsılmayı azaltmak için minimum görünür alan
        min_span = 1.0
        new_w = max(new_w, min_span)
        new_h = max(new_h, min_span)
        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
        self.ax.set_xlim(xdata - new_w * relx, xdata + new_w * (1 - relx))
        self.ax.set_ylim(ydata - new_h * rely, ydata + new_h * (1 - rely))
        self.canvas.draw_idle()

    # --- Devasa harita üretimi ve çizimi ---
    def generate(self):
        n = self.num_points.get()
        xs = self.x_spread.get()
        ys = self.y_spread.get()
        rough = self.roughness.get()

        # Büyük ölçekli ada üretimi (optimizasyon: aşırı büyük n için örneklemeyi sınırla)
        k = int(min(n, 20000))
        sample = np.random.normal(loc=[0.0, 0.0], scale=[xs, ys], size=(k, 2))
        # Yaklaşık dış sınır: örneklem üzerinden min/max hesapla
        min_x, max_x = float(np.min(sample[:,0])), float(np.max(sample[:,0]))
        min_y, max_y = float(np.min(sample[:,1])), float(np.max(sample[:,1]))
        # Başlangıç kaba çokgen: dikdörtgen
        rect = [
            (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y), (min_x, min_y)
        ]
        # Pürüzlendirme: tek sefer subdivide, aşırı büyümeyi önle
        iterations = int(rough) + 2
        poly = self._subdivide(rect, iterations, displacement_factor=0.25)
        # Kapat
        if poly[0] != poly[-1]:
            poly.append(poly[0])

        self.current_points = np.array(poly)
        self.current_settings = {
            'num_points': n,
            'x_spread': xs,
            'y_spread': ys,
            'roughness': rough
        }
        self._draw(self.current_points)

    def _subdivide(self, points, iterations, displacement_factor=0.3):
        if iterations <= 0:
            return points
        new_points = []
        rng = np.random.default_rng()
        for i in range(len(points) - 1):
            p1 = np.array(points[i])
            p2 = np.array(points[i+1])
            mid = (p1 + p2) / 2.0
            # displacement length ~ edge length
            edge = p2 - p1
            elen = np.linalg.norm(edge)
            if elen == 0:
                disp = np.array([0.0, 0.0])
            else:
                normal = np.array([-edge[1], edge[0]]) / elen
                mag = (rng.random() - 0.5) * elen * displacement_factor
                disp = normal * mag
            new_points.append(tuple(p1))
            new_points.append(tuple(mid + disp))
        new_points.append(tuple(points[-1]))
        return self._subdivide(new_points, iterations - 1, displacement_factor)

    def _draw(self, points):
        # Patch cache ile yeniden çizimi hafiflet
        if points is None:
            self.ax.clear()
            self.ax.set_facecolor('#6CA0DC')
            self.ax.axis('off')
            self.canvas.draw_idle()
            return

        if self.poly_patch is None:
            self.ax.clear()
            self.ax.axis('equal')
            self.ax.axis('off')
            self.ax.set_facecolor('#6CA0DC')
            self.poly_patch = Polygon(points, closed=True, facecolor="#88c0d0", edgecolor="#2e3440", linewidth=1.0, antialiased=False)
            self.ax.add_patch(self.poly_patch)
        else:
            self.poly_patch.set_xy(points)

        xs = points[:,0]
        ys = points[:,1]
        pad_x = (xs.max() - xs.min()) * 0.05 + 1.0
        pad_y = (ys.max() - ys.min()) * 0.05 + 1.0
        self.ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
        self.ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axis('off')
        self.canvas.draw_idle()

        # removed duplicate drawing block

    # --- Kayıt işlemleri ---
    def save_current(self):
        if self.current_points is None:
            messagebox.showwarning("Uyarı", "Önce bir harita oluşturun.")
            return
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Geçersiz İsim", "Lütfen bir kayıt adı girin.")
            return
        self.saved_maps[name] = {
            'points': np.array(self.current_points).tolist(),
            'settings': dict(self.current_settings)
        }
        self._persist()
        self._refresh_list()
        messagebox.showinfo("Başarılı", f"'{name}' kaydedildi.")

    def _persist(self):
        with open(self.saves_file, 'w') as f:
            json.dump(self.saved_maps, f, indent=4)

    def load_saves(self):
        try:
            with open(self.saves_file, 'r') as f:
                self.saved_maps = json.load(f)
                self._refresh_list()
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            messagebox.showerror("Hata", f"{self.saves_file} dosyası bozuk veya boş.")

    def _refresh_list(self):
        # Sol listedeki isimleri yenile
        # Liste yoksa çık
        for child in self.control_frame.winfo_children():
            if isinstance(child, tk.Listbox):
                child.delete(0, tk.END)
                for name in sorted(self.saved_maps.keys()):
                    child.insert(tk.END, name)
                break

    def on_select(self, event):
        sel = self.listbox.curselection()
        if not sel:
            return
        name = self.listbox.get(sel[0])
        item = self.saved_maps.get(name)
        if not item:
            return
        points = np.array(item.get('points'))
        settings = item.get('settings', {})
        self.current_points = points
        self.current_settings = settings
        self._draw(points)

if __name__ == "__main__":
    root = tk.Tk()
    app = HugeMapViewer(root)
    root.mainloop()