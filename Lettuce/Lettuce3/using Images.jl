using Images
using Colors
using Plots
img = load("./LightMaps.png")

# 定义新的宽度和高度
new_width = 800
new_height = 600
dpi =3000 
# 调整图像的大小
plot(img, size=(new_width, new_height), dpi=dpi)
resized_img = imresize(img, new_width, new_height)

# 保存调整后的图像
savefig("ResizedImage.png")
