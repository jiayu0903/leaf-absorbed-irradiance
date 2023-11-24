using Colors
using Images

img1 = load("./light map dark green/lightmap.png")
img2 = load("./light map light green/lightmap.png")

img3 = similar(img1)

m, n = size(img1)
for j in 1:n
    for i in 1:m
        img3[i,j] = img1[i,j].r > img2[i,j].r ? img1[i,j] : img2[i,j]
    end
end

img3

save("LightMaps.png", img3)