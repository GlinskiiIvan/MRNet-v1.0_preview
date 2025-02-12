import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ipywidgets import interact, Dropdown, IntSlider
# %matplotlib notebook
plt.ion()
plt.style.use('grayscale')

from pathlib import Path

# data_path = Path('C:/Users/brodyga/Documents/МРТ проект/data/MRNet-v1.0')   # Укажи путь к папке с CSV-файлом
# train_path = Path('C:/Users/brodyga/Documents/МРТ проект/data/MRNet-v1.0/train')  # Укажи путь к папке с MRI-сканами
data_path = Path(input("Укажи путь к папке с CSV-файлом: "))
train_path = Path(input("Укажи путь к папке с MRI-сканами (npy): "))

train_abnl = pd.read_csv(data_path/'train-abnormal.csv', header=None,
                       names=['Case', 'Abnormal'], 
                       dtype={'Case': str, 'Abnormal': np.int64})

# data loading functions
def load_one_stack(case, data_path=train_path, plane='coronal'):
    fpath = data_path/plane/'{}.npy'.format(case)
    return np.load(fpath)

def load_stacks(case, data_path=train_path):
    x = {}
    planes = ['coronal', 'sagittal', 'axial']
    for i, plane in enumerate(planes):
        x[plane] = load_one_stack(case, data_path=data_path, plane=plane)
    return x

class KneePlot():
    def __init__(self, x, plane):
        self.x = x
        self.planes = list(x.keys())
        self.current_plane = plane
        self.current_slice = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.25)
        self.im = self.ax.imshow(self.x[self.current_plane][self.current_slice, :, :], cmap='gray')
        
        # Ползунок для срезов
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(ax_slider, 'Slice', 0, self.x[self.current_plane].shape[0] - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update)
        
        plt.show(block=True)

    def update(self, val):
        self.current_slice = int(self.slider.val)
        self.im.set_data(self.x[self.current_plane][self.current_slice, :, :])
        self.fig.canvas.draw_idle()

# Запрос выбора плоскости у пользователя
planes_dict = {'1': 'coronal', '2': 'sagittal', '3': 'axial'}
print("Выберите плоскость для отображения:")
print("1 - Coronal (фронтальный)")
print("2 - Sagittal (сагиттальный)")
print("3 - Axial (аксиальный)")

while True:
    plane_choice = input("Введите номер плоскости (1/2/3): ")
    if plane_choice in planes_dict:
        selected_plane = planes_dict[plane_choice]
        break
    else:
        print("Ошибка! Введите 1, 2 или 3.")

# Загрузка данных и запуск визуализации
case = train_abnl.Case[0]
x = load_stacks(case)

plot = KneePlot(x, selected_plane)  # Передаем выбранную плоскость