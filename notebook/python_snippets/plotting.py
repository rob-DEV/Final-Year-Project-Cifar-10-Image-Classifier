
# x = [1 * i for i in range(len(histoR))]

# plt.figure()
# plt.subplot(3,2, (1,3))
# plt.imshow(pixels)

# axR = plt.subplot(3,2,2)
# plt.text(0.02, 0.85, "R", fontweight="bold", transform=axR.transAxes)
# plt.plot(x, histoR, color='red')
# plt.fill_between(x, histoR, color='red')

# axG = plt.subplot(3,2,4)
# plt.text(0.02, 0.85, "G", fontweight="bold", transform=axG.transAxes)
# plt.plot(x, histoG, color='green')
# plt.fill_between(x, histoG, color='green')

# axB = plt.subplot(3,2,6)
# plt.text(0.02, 0.85, "B", fontweight="bold", transform=axB.transAxes)
# plt.plot(x, histoB, color='blue')
# plt.fill_between(x, histoB, color='blue')

# y_scale = max(histoR + histoB + histoG) * 1.2

# axR.set_ylim([0, y_scale])
# axG.set_ylim([0, y_scale])
# axB.set_ylim([0, y_scale])

# plt.show()

# t0 = time.time()
# for img in images_with_classification:
#     channel_histograms = ColorHistogram(target_color_ranges=[TargetColorRange.R, TargetColorRange.G, TargetColorRange.B])
#     channel_histograms.process_image(img)
#     channel_histograms.save_to_directory(img, "data\\pre_processed\\color_histogram")

# t1 = time.time()

# color_histograms = ColorHistogram(target_color_ranges=[TargetColorRange.R, TargetColorRange.G, TargetColorRange.B])
#         color_histograms.process_image(img)

#         histoR = color_histograms.histograms[0]

#         x = [1 * i for i in range(histoR.size)]
#         axR = plt.subplot(3,2,2)
#         plt.text(0.02, 0.85, "R", fontweight="bold", transform=axR.transAxes)
#         plt.plot(x, histoR, color='red')
#         plt.fill_between(x, histoR, color='red')
#         plt.show()
