from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

source = Path(r'C:/Users/Erwin/AppData/Local/Temp/codex-clipboard-5dcb5eeb-d4a6-4f12-bbdf-e80a458ae5b8.png')
out = Path('output/efteling')
rgb = np.asarray(Image.open(source).convert('RGB')).astype(int)
mask = (rgb[:, :, 0] > 80) & (rgb[:, :, 0] > 1.6 * rgb[:, :, 1]) & (rgb[:, :, 0] > 1.3 * rgb[:, :, 2])
h, w = mask.shape
edges = {}
for y, x in zip(*np.where(mask)):
    if y == 0 or not mask[y-1, x]: edges[(x,y)] = (x+1,y)
    if x == w-1 or not mask[y, x+1]: edges[(x+1,y)] = (x+1,y+1)
    if y == h-1 or not mask[y+1, x]: edges[(x+1,y+1)] = (x,y+1)
    if x == 0 or not mask[y, x-1]: edges[(x,y+1)] = (x,y)
loops = []
while edges:
    start = next(iter(edges)); p = start; loop = [p]
    while p in edges:
        p = edges.pop(p); loop.append(p)
        if p == start: break
    if loop[-1] == start: loops.append(np.array(loop, dtype=float))
assert loops, 'No closed contours detected'
points = max(loops, key=len)
# Mild circular smoothing of pixel stair steps, preserving closed periodic geometry.
p = points[:-1]
for _ in range(4): p = (np.roll(p,1,axis=0) + 2*p + np.roll(p,-1,axis=0))/4
p[:,1] *= -1
p -= p.min(axis=0)
p *= 0.20 / np.ptp(p[:,0])
p += np.array([0.045, (0.29-np.ptp(p[:,1]))/2])
p = np.vstack([p,p[0]])
s = np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
samples = np.linspace(0,s[-1],1001)
xy = np.column_stack([np.interp(samples,s,p[:,j]) for j in range(2)])
np.savetxt(out/'efteling_contour.csv', np.column_stack([samples,xy]), delimiter=',',header='s_m,x_m,y_m',comments='',fmt='%.9f')
# Plot sampled coordinate data on equal-scale physical axes.
im=Image.new('RGB',(800,850),'white'); draw=ImageDraw.Draw(im)
def screen(q): return (70+q[0]/0.29*660,740-q[1]/0.29*660)
draw.rectangle((70,80,730,740),outline='#999999',width=2)
for tick in [0,.05,.10,.15,.20,.25,.29]:
    a,b=screen((tick,0)); draw.text((a-12,b+10),f'{tick:.2f}',fill='black')
    a,b=screen((0,tick)); draw.text((20,b-5),f'{tick:.2f}',fill='black')
draw.line([screen(q) for q in xy],fill='#b5122c',width=3)
a,b=screen(xy[0]);draw.ellipse((a-5,b-5,a+5,b+5),fill='#1565c0')
draw.text((70,25),'Efteling contour | x and y in metres | blue dot = start',fill='black')
draw.text((300,790),'x [m]',fill='black');draw.text((15,50),'y [m]',fill='black')
im.save(out/'efteling_contour.png')
print('Closed loops:',len(loops),'Points:',len(xy),'Path length [m]:',round(s[-1],4))
print('Bounds [m]:',xy.min(axis=0),xy.max(axis=0))
