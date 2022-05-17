from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']###防止中文显示不出来
plt.rcParams['axes.unicode_minus'] = False###防止坐标轴符号显示不出来
digits=load_digits()
print(digits.items())
fig,axes=plt.subplots(2,5,figsize=(10,5),subplot_kw={"xticks":(),"yticks":()})
for ax,img in zip(axes.ravel(),digits.images):
    ax.imshow(img)